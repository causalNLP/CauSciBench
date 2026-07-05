<h1 align="center">
<br>
CauSciBench: A Comprehensive Benchmark for End-to-End Causal Inference in Scientific Research
</h1>



## Key Folders

| Folder | Description |
|---|---|
| `causci_bench` | Core Python library for running baselines and evaluating model outputs |
| `data` | CSV Datasets and metadata info (in CSV and JSON format) describing the key attributes  |
| `evaluation` | Scripts for computing evaluation metrics across all models and prompting strategies |
| `reference` | Pointers to results of interest in the source papers |
| `replications` | Code to reproduce the causal analyses|
| `scripts` | Shell scripts for running baselines and evaluation |

## Overview

**CauSciBench** is a benchmark designed to evaluate end-to-end causal inference capabilities of LLMs. Closely following the causal analysis workflow, our benchmark assesses the ability of AI models to:

- Parse and understand dataset descriptions and queries
- Identify treatment and outcome variables
- Choose appropriate inference models and method-specific variables (e.g., instruments, running variables)
- Implement the selected methods
- Provide statistical interpretations of results in the context of the original query

## Benchmark Data

### Data Sources

The benchmark comprises queries from three sources:

1. **Real-world Studies**
   - Published papers on empirical causal inference from diverse disciplines including economics, political science, healthcare, and criminology

2. **Synthetic Scenarios**
   - Synthetically generated data with known causal effects
   - Hypothetical contexts and variables generated to resemble real-world causal analysis

3. **QRData-CI**
   - Examples focused on causal inference from [QRData](https://github.com/xxxiaol/QRData) (Liu et al., 2024)

## License

We use data from published papers, and the usage terms vary from dataset to dataset. Details about the licenses are provided in `data/README.md`.

Our code is provided under the MIT License.

**Important**: Users must comply with the license terms of each individual dataset they use. Always review the license terms at the original data sources and ensure compliance.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/causalNLP/CauSciBench
   cd CauSciBench
   ```

2. Installation:
   
   a. We recommend creating a new virtual environment:
   ```bash
   conda create --name causci python=3.8
   ```
   
   If you already have a virtual environment set up, you can skip this step.
   
   b. Activate the virtual environment:
   ```bash
   conda activate causci
   ```
   
   c. Install the package:
   ```bash
   pip install -e .
   ```
   
   This installs a Python library called `causci_bench`.

3. To test the installation:
   ```bash
   python -c "import causci_bench; print('Installation successful!')"
   ```

## Next Steps

### 1. Building Docker Image

We use Docker containers to run our baseline models. To set this up:

```bash
docker build -t python-baseline-http -f causci_bench/baselines/Dockerfile.http causci_bench/baselines
```

### 2. Replicating Results / Running Baselines

You can run experiments using either the provided script or directly with Python:

**Using the script:**
```bash
bash scripts/run_baseline.sh
```

**Using Python directly:**
```bash
python causci_bench/baselines/run_baselines.py \
  --queries data/metadata_json/qr_input.json \
  --output output/qrdata/qrdata_react_gpt-4o.json \
  --api openai \
  --model gpt-4o \
  --persistent \
  --react \
  --data-type qrdata
```

**Key Parameters:**
- `--queries`: Path to JSON file with causal questions
- `--output`: File path where results are saved
- `--api`: LLM provider (e.g., openai, together)
- `--model`: LLM model (e.g., gpt-4o)
- `--persistent`: Use stateful Python environment
- `--pot/--react/--chain`: Different prompting strategies; default is direct prompting
- `--data-type`: Dataset category (real, synthetic, qrdata)

#### How causci_bench/baselines/run_baselines.py Works

1. **Load queries**: Reads JSON files containing causal questions and attributes pertaining to causal inference
2. **Docker setup**: Starts Python containers for code execution
3. **Execution loop**: For each query:
   - Sends the question along with the context to the selected LLM
   - LLM generates Python code for causal estimation
   - Executes code in Docker container
   - Iterates if an error arises
   - Extracts the key results
4. **Save results**: Outputs a JSON file with chat history, code, and analysis

### 3. Running Evaluation

After collecting model outputs, compute evaluation metrics across all models and prompting strategies:

**Using the script (all three datasets):**
```bash
bash scripts/run_evaluation.sh
```

**Using Python directly (single dataset):**
```bash
python evaluation/run_evaluation.py \
  --source qrdata \
  --output_folder output/qrdata \
  --meta_path data/metadata_csv/qr_info.csv \
  --output_dir evaluation/results
```

**Key Parameters:**
- `--source`: Dataset category (`real`, `synthetic`, `qrdata`)
- `--output_folder`: Folder containing the JSON output files from `run_baselines.py`
- `--meta_path`: Path to the metadata CSV file for the dataset
- `--output_dir`: Directory where the results CSV is saved (default: `evaluation/results`)

Results are saved as `results_{source}.csv` with one row per (prompt, model) combination, reporting:
- `method_accuracy`, `method_f1`: method selection accuracy and macro F1
- `effect_accuracy`: causal effect accuracy (within 5% of the true value)
- `treatment_accuracy`, `outcome_accuracy`: variable selection accuracy
- `control_overlap`: Jaccard-style overlap of predicted vs. reference control variables
- `completion_rate`, `mean_attempts`: how often the pipeline produces a valid result and average retries

## Other Notes

Details on our approach for generating synthetic data are provided in the README file in `causci_bench/synthetic`.
