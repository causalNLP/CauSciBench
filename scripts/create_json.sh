## This script runs python programs to create the json files 

BASE_LOC="data"
QRDATA_PATH="${BASE_LOC}/qr_info.csv"
REAL_PATH="${BASE_LOC}/real_info.csv"
SYNTHETIC_PATH="${BASE_LOC}/synthetic_info.csv"

FILENAME_COL="data_files"
DESC_NAME_COL="data_description"
QUERY_NAME_COL="natural_language_query"
PAPER_NAME_COL="paper_name"

OUTPUT_FOLDER="data/json"
REAL_OUTPUT_NAME="real_data.json"
SYNTHETIC_OUTPUT_NAME="synthetic_data.json"
QR_OUTPUT_NAME="qrdata.json"


python create_json.py -i ${REAL_PATH} -of ${OUTPUT_FOLDER} -on ${REAL_OUTPUT_NAME} \
       -nc ${PAPER_NAME_COL} -qc ${QUERY_NAME_COL} -dc ${DESC_NAME_COL} \
       -fc ${FILENAME_COL} -itc "interacting_var"
       
#python create_json.py -i ${SYNTHETIC_PATH} -of ${OUTPUT_FOLDER} -on ${SYNTHETIC_OUTPUT_NAME} \
       -nc ${PAPER_NAME_COL} -qc ${QUERY_NAME_COL} -dc ${DESC_NAME_COL} \
       -fc ${FILENAME_COL}

#python create_json.py -i ${QRDATA_PATH} -of ${OUTPUT_FOLDER} -on ${QR_OUTPUT_NAME} \
       -nc ${PAPER_NAME_COL} -qc ${QUERY_NAME_COL} -dc ${DESC_NAME_COL} \
       -fc ${FILENAME_COL}
       
