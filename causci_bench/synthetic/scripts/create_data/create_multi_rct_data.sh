#!/bin/sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../settings.sh"

METHOD="multi_rct"
METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata"
DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"

python causci_bench/synthetic/generation/generate_synthetic.py -md ${METADATA_FOLDER} -d ${DATA_FOLDER} -m ${METHOD} -s ${DEFAULT_SIZE} -mb ${N_BINARY} -mc ${N_CONTINUOUS_MULTI}  -nt ${MAX_TREATMENTS} -o ${DEFAULT_OBS}
