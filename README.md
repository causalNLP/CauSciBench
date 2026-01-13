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

- To run the baseline models, see instructions in `causci_bench/baselines`
- To generate synthetic data, see `causci_bench/synthetic`

