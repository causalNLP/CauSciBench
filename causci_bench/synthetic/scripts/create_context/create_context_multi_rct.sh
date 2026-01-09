#!/bin/sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../settings.sh"

METHOD="multi_rct"
METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata/${METHOD}.json"
DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"
OUTPUT_FOLDER="${BASE_FOLDER}/${METHOD}/description"

python causci_bench/synthetic/context/generate_context.py -mp ${METADATA_FOLDER} -d ${DATA_FOLDER} -o ${OUTPUT_FOLDER} -m ${METHOD}
