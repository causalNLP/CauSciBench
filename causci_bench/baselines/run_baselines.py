import json
import os
import tqdm
import argparse
import pandas as pd
from pathlib import Path

# Ensure project root is on sys.path when running this file directly
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import baselines as base
from baselines import DirectFormat, CausalCoTFormat, ProgramOfThoughtsFormat, ReActFormat


def main(args):
    root = Path(__file__).resolve().parent
    project_root = root.parent.parent
    data_root = project_root / "data"

    # Resolve queries path relative to project root if not absolute
    queries_path = args.queries
    if not os.path.isabs(queries_path):
        resolved = project_root / queries_path
        if resolved.exists():
            queries_path = str(resolved)

    # Determine the base path for datasets
    if args.data_type == 'qrdata':
        base_path = str(data_root / 'csv_files' / 'qrdata')
    elif args.data_type == 'real':
        base_path = str(data_root / 'csv_files' / 'realdata')
    elif args.data_type == 'synthetic':
        base_path = str(data_root / 'csv_files' / 'synthetic_data')
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    # Load queries based on file type
    if queries_path.endswith('.csv'):
        df = pd.read_csv(queries_path)
        df = df.rename(columns={
            'data_description': 'dataset_description',
            'dataset_name': 'dataset_path'
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

    # Initialize the chatbot
    if args.rpc_address:
        chatbot = base.RPCChatbot(args.rpc_address)
    else:
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

    # Initialize the baseline
    model = base.Baseline(chatbot, persistent=args.persistent, session_timeout=args.session_timeout,
                          max_steps=args.max_steps)

    # Start persistent session if enabled
    if args.persistent:
        print("Starting persistent Python environment...")
        if model.start_persistent_session():
            print("Persistent environment started successfully.")
            if hasattr(chatbot, 'persistent_mode'):
                chatbot.persistent_mode = True
                print("Updated chatbot to use persistent mode.")
        else:
            print("Failed to start persistent environment. Falling back to one-off mode.")
            args.persistent = False

    # Determine query format
    qf = DirectFormat
    if args.pot:
        qf = ProgramOfThoughtsFormat
    elif args.react:
        qf = ReActFormat
    elif args.chain:
        qf = CausalCoTFormat
    elif args.chainreact:
        qf = ChainReactFormat

    if qf in (ReActFormat, ChainReactFormat) and not args.persistent:
        print(f"Warning: {qf.__name__} requires persistent mode. "
              "Code blocks may fail due to missing state from previous steps.")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output = []
    for q in tqdm.tqdm(queries, desc="Processing queries"):
        try:
            query = q["query"]
            dataset_path = q["dataset_path"]
            dataset_description = q["dataset_description"]

            # If in persistent mode, upload the dataset file to the container
            if args.persistent and os.path.exists(dataset_path):
                print(f"Uploading dataset file {dataset_path} to container...")
                container_path = dataset_path
                upload_result = model.upload_file(dataset_path, container_path)
                print(upload_result)
                print(f"Dataset uploaded to container at path: {container_path}")

            print(f"Processing query: {query[:100]}...")
            result = model.answer(query, dataset_path, dataset_description, qf=qf,
                                  post_steps=False)
            output.append({**q, "result": result})

        except Exception as e:
            print(f"Error on query '{str(q.get('query', ''))[:80]}': {e}")
            import traceback
            traceback.print_exc()
            output.append({**q, "result": None, "error": str(e)})

    # Save the output
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {args.output}")

    # Clean up persistent session if it was used
    if args.persistent:
        print("Stopping persistent Python environment...")
        model.stop_persistent_session()
        print("Persistent environment stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, default="data/metadata_json/qr_input.json",
                        help="Path to the queries file (JSON or CSV)")
    parser.add_argument("--output", type=str, default="runs/output.json",
                        help="Path to the output json file")
    parser.add_argument("--model", type=str, default="google/gemini-1.5-flash-001",
                        help="Name of the model to use")
    parser.add_argument("--data-type", type=str, default="qrdata",
                        choices=['qrdata', 'real', 'synthetic'],
                        help="Type of data to process (qrdata, real, or synthetic)")
    parser.add_argument("--api", type=str, default="azure",
                        help="Type of API to use. Options: vertex, azure, test, local, openai, together.")
    parser.add_argument("--rpc-address", type=str, default=None,
                        help="Address of the RPC server to connect to (will override the --api flag)")
    parser.add_argument("--pot", action=argparse.BooleanOptionalAction,
                        help="Use the Program of Thoughts (PoT) approach for causal analysis")
    parser.add_argument("--react", action=argparse.BooleanOptionalAction,
                        help="Use the ReAct approach for causal analysis")
    parser.add_argument("--chain", action=argparse.BooleanOptionalAction,
                        help="Use the Causal Chain of Thought (CoT) approach for causal analysis")
    parser.add_argument("--chainreact", action=argparse.BooleanOptionalAction,
                        help="Use the Chain of Thought with ReAct (ChainReact) for causal analysis")
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction,
                        help="Use persistent Python environment for code execution")
    parser.add_argument("--session-timeout", type=int, default=3600,
                        help="Timeout for persistent sessions in seconds (default: 3600)")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Maximum number of ReAct steps for ReAct/ChainReact formats (default: 15)")

    args = parser.parse_args()
    main(args)
