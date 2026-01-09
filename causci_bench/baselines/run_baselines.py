import json
import os
import tqdm
import argparse
import pandas as pd
import multiprocessing as mp
from functools import partial
import time
from pathlib import Path

# Ensure project root is on sys.path when running this file directly
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import baselines as base
from baselines import CausalQueryFormat, CausalCoTFormat
from baselines.query_formats import ProgramOfThoughtsFormat, ReActFormat


def process_single_query(query_data, args, worker_id):
    """Process a single query in a worker process."""
    try:
        # Initialize chatbot for this worker
        if args.rpc_address:
            chatbot = base.RPCChatbot(args.rpc_address)
        else:
            # Initialize the chatbot with worker-specific settings if needed
            if args.api == "test":
                chatbot = base.TestChatbot()
            elif args.api == "vertex":
                chatbot = base.VertexAPIChatbot(model=args.model, persistent_mode=args.persistent)
            elif args.api == "azure":
                chatbot = base.AzureAPIChatbot(model=args.model, persistent_mode=args.persistent)
            elif args.api == "openai":
                chatbot = base.OpenAIAPIChatbot(model=args.model, persistent_mode=args.persistent)
            elif args.api == "together":
                chatbot = base.TogetherAPIChatbot(model=args.model, persistent_mode=args.persistent)
            elif args.api == "local":
                raise NotImplementedError("Local chatbot is not implemented yet.")
            else:
                raise ValueError(f"Invalid API: {args.api}")

        # Initialize the baseline with worker-specific ID
        model = base.Baseline(chatbot, persistent=args.persistent, 
                                             session_timeout=args.session_timeout, worker_id=worker_id)

        # Initialize persistent_mode first
        persistent_mode = args.persistent
        
        # Start persistent session if enabled
        if args.persistent:
            print(f"Worker {worker_id}: Starting persistent Python environment...")
            if model.start_persistent_session():
                print(f"Worker {worker_id}: Persistent environment started successfully.")
                
                # If using persistent mode, update the chatbot's system message
                if hasattr(chatbot, 'persistent_mode'):
                    chatbot.persistent_mode = True
                    print(f"Worker {worker_id}: Updated chatbot to use persistent mode.")
                
                persistent_mode = True
            else:
                print(f"Worker {worker_id}: Failed to start persistent environment. Falling back to one-off mode.")
                persistent_mode = False

        query = query_data["query"]
        dataset_path = query_data["dataset_path"]
        dataset_description = query_data["dataset_description"]
        
        # If in persistent mode, upload the dataset file to the container
        if persistent_mode and os.path.exists(dataset_path):
            print(f"Worker {worker_id}: Uploading dataset file {dataset_path} to container...")
            # Use the same path structure in the container as the original path
            container_path = dataset_path
            upload_result = model.upload_file(dataset_path, container_path)
            print(f"Worker {worker_id}: {upload_result}")
            
            # No need to update the dataset path as we're using the same path structure
            print(f"Worker {worker_id}: Dataset uploaded to container at path: {container_path}")

        # Determine query format
        qf = CausalQueryFormat
        if args.potm:
            qf = ProgramOfThoughtsFormat
        if args.react:
            qf = ReActFormat
        if args.chain:
            qf = CausalCoTFormat

        print(f"Worker {worker_id}: Processing query: {query[:100]}...")
        
        # Process the query
        result = model.answer(query, dataset_path, dataset_description, qf=qf, post_steps=False)

        # Clean up persistent session if it was used
        if persistent_mode:
            print(f"Worker {worker_id}: Stopping persistent Python environment...")
            model.stop_persistent_session()
            print(f"Worker {worker_id}: Persistent environment stopped.")

        return {
            **query_data,
            "result": result,
            "worker_id": worker_id,
            "status": "success"
        }

    except Exception as e:
        print(f"Worker {worker_id}: Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            if 'scientist' in locals() and args.persistent:
                model.stop_persistent_session()
        except:
            pass
            
        return {
            **query_data,
            "result": None,
            "worker_id": worker_id,
            "status": "error",
            "error": str(e)
        }


def collect_results_with_progress(pool_results, total_queries):
    """Collect results from the pool with a progress bar."""
    results = []
    
    # Create a progress bar
    pbar = tqdm.tqdm(total=total_queries, desc="Processing queries")
    
    # Collect results as they complete
    for result in pool_results:
        try:
            # This will block until the result is ready
            completed_result = result.get()
            results.append(completed_result)
            pbar.update(1)
            
            # Print status
            if completed_result["status"] == "success":
                pbar.set_description(f"Completed (Worker {completed_result['worker_id']})")
            else:
                pbar.set_description(f"Error (Worker {completed_result['worker_id']})")
                
        except Exception as e:
            print(f"Error collecting result: {e}")
            pbar.update(1)
    
    pbar.close()
    return results


def main(args):
    queries_path = args.queries
    root = Path(__file__).resolve().parent
    data_root = root.parent / "data"


    # Determine the base path for datasets
    if args.data_type == 'qrdata':
        base_path = str(data_root / 'qrdata')
    elif args.data_type == 'real':
        base_path = str(data_root / 'real_data')
    elif args.data_type == 'synthetic':
        base_path = str(data_root / 'synthetic_data')
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    # Load queries based on file type
    if queries_path.endswith('.csv'):
        df = pd.read_csv(queries_path)
        # Rename columns to match the expected format
        df = df.rename(columns={
            'natural_language_query': 'query',
            'data_description': 'dataset_description',
            'data_files': 'dataset_path'
        })
        queries = df.to_dict('records')
    elif queries_path.endswith('.json'):
        with open(queries_path, "r") as f:
            print(f"Loading queries from {queries_path}")
            queries = json.load(f)
    else:
        raise ValueError("Unsupported file type for --queries. Please use .csv or .json")

    # Unify dataset path construction
    for q in queries:
        filename = os.path.basename(q['dataset_path'])
        q['dataset_path'] = os.path.join(base_path, filename)

    print(f"Loaded {len(queries)} queries")
    print(f"Using {args.num_workers} workers for parallel processing")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process queries in parallel
    start_time = time.time()
    
    if args.num_workers == 1:
        # Sequential processing (for debugging or comparison)
        print("Running in sequential mode...")
        results = []
        for i, query in enumerate(tqdm.tqdm(queries, desc="Processing queries")):
            result = process_single_query(query, args, worker_id=0)
            results.append(result)
    else:
        # Parallel processing
        print(f"Running in parallel mode with {args.num_workers} workers...")
        
        # Create a pool of worker processes
        with mp.Pool(processes=args.num_workers) as pool:
            # Submit all jobs to the pool
            pool_results = []
            for i, query in enumerate(queries):
                # Assign worker ID based on query index
                worker_id = i % args.num_workers
                result = pool.apply_async(process_single_query, (query, args, worker_id))
                pool_results.append(result)
            
            # Collect results with progress tracking
            results = collect_results_with_progress(pool_results, len(queries))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Report statistics
    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "error"]
    
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per query: {total_time/len(queries):.2f} seconds")
    print(f"Successful queries: {len(successful_results)}")
    print(f"Failed queries: {len(failed_results)}")
    
    if failed_results:
        print("\nFailed queries:")
        for result in failed_results:
            print(f"  - Query: {result['query'][:50]}... (Error: {result.get('error', 'Unknown')})")

    # Save the output
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_baselines_parallel.py", description="Run the baselines with parallel processing",
        epilog="Example: python run_baselines_parallel.py --queries queries/queries.json --output runs/output.json --model google/gemini-1.5-flash-002 --num-workers 4")
    parser.add_argument("--queries", type=str, default="benchmark/qrdata/qrdata_info.csv", 
                        help="Path to the queries file (JSON or CSV)")
    parser.add_argument("--output", type=str, default="runs/output.json",
                        help="Path to the output json file")
    parser.add_argument("--model", type=str,default="google/gemini-1.5-flash-001",
                        help="Name of the model to use")
    parser.add_argument("--query-format", type=str, default="CausalQueryFormat",
                        help="Name of the QueryFormat class to use")
    parser.add_argument("--data-type", type=str, default="qrdata",
                        choices=['qrdata', 'real', 'synthetic'], help="Type of data to process (qrdata, real, or synthetic)")
    parser.add_argument("--api", type=str, default="azure",
        help="Type of API to use. Options: vertex, azure, test, local, openai, together. Choosing 'local' will use a local chatbot.")
    parser.add_argument("--rpc-address", type=str, default=None,
        help="Address of the RPC server to connect to (will override the --api flag)")
    parser.add_argument("--veridical", action=argparse.BooleanOptionalAction, 
                        help="Use the veridical prompting method")
    parser.add_argument("--potm", action=argparse.BooleanOptionalAction, 
                        help="Use the program of thoughts approach for causal analysis")
    parser.add_argument("--react", action=argparse.BooleanOptionalAction, 
                        help="Use the ReAct approach for causal analysis")
    parser.add_argument("--chain", action=argparse.BooleanOptionalAction, 
                        help="Use the Causal Chain of Thought (CausalCoT) for causal analysis")
    parser.add_argument("--method-explanation", action=argparse.BooleanOptionalAction, 
                        help="(For the baseline) Use method explanation")
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction, 
                        help="Use persistent Python environment for code execution")
    parser.add_argument("--session-timeout", type=int, default=3600, 
                        help="Timeout for persistent sessions in seconds (default: 3600)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help=f"Number of worker processes to use (default: {mp.cpu_count()})")

    args = parser.parse_args()

    # Validate number of workers
    if args.num_workers < 1:
        args.num_workers = 1
    elif args.num_workers > mp.cpu_count():
        print(f"Warning: Requested {args.num_workers} workers, but only {mp.cpu_count()} CPUs available")

    main(args)
