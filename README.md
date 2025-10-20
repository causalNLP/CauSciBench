<h1 align="center">
<br>
CAUSCIBENCH: A COMPREHENSIVE BENCHMARK ON
END-TO-END CAUSAL INFERENCE FOR SCIENTIFIC RESEARCH
</h1>

**Note**: This is a work in progress. We will update the repository frequently in the subsequent days.

## Overview

CauSciBench is a benchmark dataset designed to evaluate the ability of AI tools to perform end-to-end causal inference. Closely following the scientific causal analysis workflow, our dataset aims to assess the ability of models to parse and understand the dataset description and the query, identify the treatment and outcome variables, choose the appropriate inference model and method-specific variables (e.g., instrument, running variable), implement it, and provide a statistical interpretation of the result in the context of the original query. To this end, our annotation consists of:

1. Description of the dataset
2. Causal query (in plain language) that does not state what method or variables to pick
3. Reference causal method
4. Causal effect estimate
5. Standard error
6. Statistical significance
7. Treatment variable
8. Outcome variable
9. Control variables / observed confounders
10. Model-specific variables including instrument (IV), running variable (RDD), time variable (DiD), state variable (DiD)

## Sources

We draw examples from three sources:

1. **Published real-world studies**: We turn to published papers on empirical causal inference from a wide range of disciplines including economics, political science, healthcare, criminology, etc. The causal inference method used in the paper and its corresponding answer is set as the reference method and causal effect. Information on the studies we have selected can be found in `data/source_info.pdf`.
2. **Textbook examples**: These examples were picked from QRData (Liu et al., 2024).
3. **Synthetic scenarios**: We generate synthetic data with known causal effects. To resemble causal analysis in practice, we generate hypothetical contexts and variables for the synthetic data using GPT.

## Data Location

The annotated data is provided in JSON format and can be found in the folder: `data/json`. Likewise, the corresponding CSV files are present in the following folders: `data/real_data`, `data/synthetic_data`, `data/qrdata`.

Additionally, a more detailed annotation is provided in three CSV files: `real_info.csv`, `qr_info.csv`, and `synthetic_info.csv`. Besides key variables, the files include links to the source, license information, modifications, and other remarks about the data.

## License

We use data from published papers, and the usage terms vary from dataset to dataset. Details about the licenses are provided in the readme.md file in each dataset folder. They can be found in the folders: `data/real_data`, `data/synthetic_data`, and `data/qrdata`. Please review the information as well as the original license terms, and ensure compliance.

**Important**: Users must comply with the license terms of each individual dataset they use. Always review the terms of use at the original data sources.

