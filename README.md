<h1 align="center">
<br>
CauSciBench: A Comprehensive Benchmark for End-to-End Causal Inference in Scientific Research
</h1>

**Note**: This is a work in progress. We will update the repository frequently in the subsequent days.

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
   - Information on selected studies can be found in `data/source_info.pdf`

2. **Synthetic Scenarios**
   - Synthetically generated data with known causal effects
   - Hypothetical contexts and variables generated to resemble real-world causal analysis

3. **Textbook Examples**
   - Examples focused on causal inference from [QRData](https://github.com/xxxiaol/QRData) (Liu et al., 2024)

## Organization of the Folder

1. `causci_bench`: associated Python library
2. `data`: folder containing our data

## License

We use data from published papers, and the usage terms vary from dataset to dataset. Details about the licenses are provided in the `README.md` file in each dataset folder. They can be found in the folders: `data/real_data`, `data/synthetic_data`, and `data/qrdata`.

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
docker build -t python-baseline-http -f baselines/Dockerfile.http baselines
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
  --queries data/json/qrdata.json \
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
- `--potm/--react/--chain`: Different prompting strategies; default is direct prompting
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

## Other Notes

Details on our approach for generating synthetic data are provided in the README file in `causci_bench/synthetic`.
