#!/bin/sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../settings.sh"

METHOD="did_canonical"
METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata"
DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"

python causci_bench/synthetic/generation/generate_synthetic.py -md ${METADATA_FOLDER} -d ${DATA_FOLDER} -m ${METHOD} -s ${DEFAULT_SIZE} -mb ${N_BINARY_OTHERS} -mc ${N_CONTINUOUS_DID_CANONICAL} -o ${DEFAULT_OBS}
