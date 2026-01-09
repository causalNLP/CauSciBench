#!/bin/sh

echo "Generating RCT Data"
bash causci_bench/synthetic/scripts/create_data/create_rct_data.sh

echo "Generating Multi-RCT Data"
bash causci_bench/synthetic/scripts/create_data/create_multi_rct_data.sh

echo "Generating Front_Door Data"
bash causci_bench/synthetic/scripts/create_data/create_front_door_data.sh

echo "Generating Observational Data"
bash causci_bench/synthetic/scripts/create_data/create_observational_data.sh

echo "Generating Canonical DiD Data"
bash causci_bench/synthetic/scripts/create_data/create_did_canonical_data.sh

echo "Generating TWFE DiD Data"
bash causci_bench/synthetic/scripts/create_data/create_did_twfe_data.sh

echo "Generating IV Data"
bash causci_bench/synthetic/scripts/create_data/create_iv_data.sh

echo "Generating IV-Encouragement Data"
bash causci_bench/synthetic/scripts/create_data/create_iv_encouragement_data.sh

echo "Generating RDD Data"
bash causci_bench/synthetic/scripts/create_data/create_rdd_data.sh

