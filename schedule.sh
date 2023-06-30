#!/bin/bash

# List of dataset names
datasets=(
"ArticularyWordRecognition"
"AtrialFibrillation"
"BasicMotions"
"Cricket"
"DuckDuckGeese"
"EigenWorms"
"Epilepsy"
"EthanolConcentration"
"ERing"
"FaceDetection"
"FingerMovements"
"HandMovementDirection"
"Handwriting"
"Heartbeat"
"Libras"
"LSST"
"MotorImagery"
"NATOPS"
"PenDigits"
"PEMS-SF"
"PhonemeSpectra"
"RacketSports"
"SelfRegulationSCP1"
"SelfRegulationSCP2"
"StandWalkJump"
"UWaveGestureLibrary"
)

# Define experiment name prefix
prefix="BASELINE-3HR-FIX3"

# List of GPUs
gpus=(0 1 2 3)

# Create a token file for each GPU with one token
for gpu in "${gpus[@]}"; do
    echo "token" > "gpu_${gpu}_tokens.txt"
done



run_experiment() {
    gpu=$1
    json_string=$2

    # Define a cleanup function
    cleanup() {
        # Add token back to the gpu tokens file
        echo "token" >> "gpu_${gpu}_tokens.txt"
        # Delete the temp file
        rm -f $tmpfile
    }

    # Set trap to ensure cleanup function is called on script exit
    trap cleanup EXIT

    # Write the JSON string to a temp file
    tmpfile="tmp.$$.$gpu.json"
    echo "$json_string" > $tmpfile

    # Run the Python script
    python main.py $tmpfile
}

# Start experiments on each GPU
for dataset in "${datasets[@]}"; do
    # Wait for a gpu to become available
    while true; do
        for gpu in "${gpus[@]}"; do
            if [[ -s "gpu_${gpu}_tokens.txt" ]]; then
                # Take a token from the gpu tokens file
                sed -i '1d' "gpu_${gpu}_tokens.txt"
                
                # Update experiment name and dataset name in the JSON file
                json_string=$(jq --argjson gpu $gpu --arg experiment_name "${prefix}-${dataset}" --arg dataset_name "$dataset" \
                    '.EXPERIMENT_NAME = $experiment_name | .WORKER_CONFIG.DATASET_CONFIG.NAME = $dataset_name | .SEARCH_CONFIG.DEVICES[0] = $gpu' config.json)
                echo "RUNNING DATASET: $dataset" 
                run_experiment "$gpu" "$json_string" &
                break 2
            fi
        done
        sleep 1
    done
done

# Wait for all background processes to finish
wait
