# Synthetic Data Generation Instructions

## Prerequisites

Ensure the `causci_bench` package is installed:
```bash
cd CauSciBench
pip install -e .
```

## Step 1: Configure Parameters

1. Navigate to the scripts directory:
   ```bash
   cd causci_bench/synthetic/scripts
   ```
2. Open `settings.sh` and configure the hyperparameters for your synthetic data generation

## Step 2: Generate Synthetic Data

### For a Single Method

Navigate to the project root directory (CauSciBench) and run the appropriate script:

```bash
# Example: Generate RCT data
bash causci_bench/synthetic/scripts/create_data/create_rct_data.sh

# Example: Generate observational data
bash causci_bench/synthetic/scripts/create_data/create_observational_data.sh
```

**Available methods:**
- `create_observational_data.sh`
- `create_rct_data.sh`
- `create_multi_rct_data.sh`
- `create_did_canonical_data.sh`
- `create_did_twfe_data.sh`
- `create_iv_data.sh`
- `create_encouragement_data.sh`
- `create_rdd_data.sh`
- `create_frontdoor_data.sh`

**Output:**

***Note:*** The results are described with respect to the default parameters in `settings.sh`. They may vary if the names are modified.

- Datasets will be saved to: `output/synthetic/{method}/data/`
- A metadata file will be created at: `output/synthetic/{method}/metadata/{method}.json`
- The metadata file contains:
  - True effects
  - Number of observations
  - Number of continuous covariates
  - Number of binary covariates

### For All Methods

To generate synthetic data for all methods in one go:
```bash
bash causci_bench/synthetic/scripts/create_synthetic_data_all.sh
```

## Step 3: Generate Contextual Information

### For a Single Method

From the project root directory, run:

```bash
# Example: Generate context for RCT data
bash causci_bench/synthetic/scripts/create_context/create_context_rct.sh
```

**What this does:**
- Generates realistic variable names (e.g., "education_years" instead of "X1")
- Creates a backstory for the dataset
- Generates a natural language causal query

**Output:**
- GPT-generated information will be saved to: `output/synthetic/{method}/description/{method}.json`

### For All Methods

To generate contextual information for all methods at once:
```bash
bash causci_bench/synthetic/scripts/create_context_all.sh
```

## Step 4: Finalize and Create Summary Files

From the project root directory, run:
```bash
bash causci_bench/synthetic/scripts/finalize_synthetic_dataset.sh
```

### Output Files

This script generates two types of output files:

1. **Summary Info Files**
   - Contains all information needed for analysis (data description, query, true effects, etc.)
   - A separate CSV file is created for each method (e.g., `rct_info.csv`)
   - Files are saved to: `output/synthetic/data_info/`

2. **Renamed Dataset Files**
   - Original columns (X1, X2, ..., Y, D) are renamed with real-world variable names
   - Generated using GPT output from Step 3
   - Files are saved to: `output/synthetic/synthetic_data/`

## Directory Structure

```
CauSciBench/
├── causci_bench/
│   └── synthetic/
│       ├── generation/          # Data generation modules
│       ├── context/             # Context generation modules
│       ├── processing/          # Data finalization modules
│       └── scripts/
│           ├── settings.sh      # Configuration parameters
│           ├── create_data/     # Data generation scripts
│           ├── create_context/  # Context generation scripts
│           └── finalize_synthetic_dataset.sh
├── output/                      # Main output directory (created automatically)
│   └── synthetic/
│       ├── data_info/           # Summary CSV files for each method
│       ├── synthetic_data/      # Renamed datasets with real variable names
│       ├── observational/
│       │   ├── data/            # Raw generated data
│       │   ├── metadata/        # Metadata JSON
│       │   └── description/     # GPT-generated context
│       ├── rct/
│       │   ├── data/
│       │   ├── metadata/
│       │   └── description/
│       ├── did_canonical/
│       ├── did_twfe/
│       ├── iv/
│       ├── iv_encouragement/
│       ├── rdd/
│       ├── multi_rct/
│       └── frontdoor/
└── logs/                        # Log files (created automatically)
    ├── log_config.ini
    └── records/
```

## Running Scripts

All scripts should be run from the project root directory (CauSciBench/) using `bash`:

```bash
# Run any script with bash command
bash causci_bench/synthetic/scripts/create_data/create_observational_data.sh
bash causci_bench/synthetic/scripts/create_context/create_context_rct.sh
bash causci_bench/synthetic/scripts/finalize_synthetic_dataset.sh
```

## Sample Results

Example outputs can be found in the `output/synthetic/` directory after running the scripts.

