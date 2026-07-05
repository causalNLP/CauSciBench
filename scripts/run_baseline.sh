## This script runs the baseline prompting approaches 


QUERIES_JSON_PATH="data/metadata_json/qr_input.json"
PROMPT="react"
API="openai"
MODEL="gpt-4o-mini"
OUTPUT_FOLDER="output/qrdata"
SOURCE="qrdata"
OUTPUT_NAME="${OUTPUT_FOLDER}/${SOURCE}_${PROMPT}_${MODEL}.json"

echo "Running model ${MODEL}"
python causci_bench/baselines/run_baselines.py --queries ${QUERIES_JSON_PATH} \
    --output ${OUTPUT_NAME} --api ${API} --model "${MODEL}" \
    --persistent --react --data-type ${SOURCE}

