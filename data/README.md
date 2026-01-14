# Data

The folder contains the datasets used in our benchmark. It includes both the csv files and their corresponding metadata.

```
├── synthetic_data/         # csv files for the synthetic data
├── qrdata/                 # csv files for QRData
├── real_data/              # csv files for real-world data files
├── metadata_json/          # the annotation in JSON format
│   ├── qrdata.json
│   ├── real_data.json
│   └── synthetic_data.json
├── metadata_csv/           # the annotation + metadata in csv file
│   ├── qrdata.csv
│   ├── real_data.csv
│   └── synthetic_data.csv
└── source_info.pdf         # details about the sources for real-world datasets
```

The JSON files are derived from the master csv file. They include only the set of information we need when running evaluations on the data. For more comprehensive information, see the csv files (in metadata_csv).

## Annotation Details

We annotate the following details pertaining to causality:

1. Description of the dataset, including the variable definition, the source, and the mechanism of collection
2. The causal query associated with the dataset
3. Reference causal method: the preferred method for causal analysis
4. Causal effect estimate
5. Standard error
6. Statistical significance
7. Treatment variable
8. Outcome variable
9. Control variables / observed confounders
10. Whether or not the data is from a randomized trial
11. Model-specific variables including instrument (for IV), running variable (for RDD), time variable (for DiD), state variable (for DiD)

## Metadata

The metadata also consists of the following information:

1. Paper name (the source paper associated with the dataset)
2. Datafile name
3. Reference table / figure in the original study related to the query
4. Publication year
5. Domain

## License

The usage license varies across the files. Please see the README in the respective folders (synthetic_data, qrdata, real_data) for details about the licenses. Concise information on the source and license for real-world datasets is provided in `source_info.pdf`.
