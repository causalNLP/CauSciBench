## Runs the evaluation for a given dataset and saves results as a CSV file

import os
import argparse
from evaluate import evaluate_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True,
                        choices=["real", "qrdata", "synthetic"],
                        help="Dataset source to evaluate")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to folder containing JSON output files")
    parser.add_argument("--meta_path", type=str, required=True,
                        help="Path to the metadata CSV file")
    parser.add_argument("--output_dir", type=str, default="evaluation/results",
                        help="Directory to save the results CSV (default: evaluation/results)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Evaluating: {args.source}")
    df = evaluate_all(args.output_folder, args.meta_path, args.source)

    output_path = os.path.join(args.output_dir, f"results_{args.source}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
