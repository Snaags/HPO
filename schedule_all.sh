#!/bin/bash

# List of dataset names
datasets=(
"NATOPS"
"RacketSports"
"FingerMovements"
"SelfRegulationSCP1"
"SelfRegulationSCP2"
"DuckDuckGeese"
"EigenWorms"
"Epilepsy"
"EthanolConcentration"
"ERing"
"FaceDetection"
"HandMovementDirection"
"Handwriting"
"Heartbeat"
"Libras"
"LSST"
"MotorImagery"
"PenDigits"
"PhonemeSpectra"
"PEMS-SF"
"StandWalkJump"
"UWaveGestureLibrary"
"ArticularyWordRecognition"
"AtrialFibrillation"
"BasicMotions"
"Cricket"
)

# Define experiment name prefix
prefix="PBT-2"

run_experiment() {
    json_string=$1

    # Write the JSON string to a temp file
    tmpfile="tmp.$$.$RANDOM.json"
    echo "$json_string" > $tmpfile

    # Run the Python script
    python main.py $tmpfile

    # Delete the temp file after the script is done
    rm -f $tmpfile
}

# Start experiments for each dataset
for dataset in "${datasets[@]}"; do
    # Update experiment name and dataset name in the JSON file
    json_string=$(jq --arg experiment_name "${prefix}-${dataset}" --arg dataset_name "$dataset" \
        '.EXPERIMENT_NAME = $experiment_name | .WORKER_CONFIG.DATASET_CONFIG.NAME = $dataset_name' config.json)
    echo "RUNNING DATASET: $dataset" 

    # Execute the Python script for the current dataset and wait for completion
    run_experiment "$json_string"
done
