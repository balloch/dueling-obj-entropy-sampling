#!/bin/bash

cd /srv/essa-lab/flash3/jballoch6/code/does

source ~/.bashrc


OUTPUT_DIR="$1"
CONFIG_FILE="$2"  ## expected naming convention is task_noveltyarg_method, like walker_175_curious-replay
SEEDS=${3:-4}
PRETRAINED_MODEL="$4"


echo "OUTPUT DIR: $OUTPUT_DIR"
echo "CONFIG FILE: $CONFIG_FILE"

# Extract the file extension
file_extension="${CONFIG_FILE##*.}"

# Set TRUE for debugging
export PYTHONUNBUFFERED=FALSE 

conda activate does

for SEED in $(seq 0 $SEEDS); do
    echo "      ---- RUN SEED $SEED ----"
    if [ -n "$4" ]; then
        PRETRAINED_MODEL="$4"
        cp -R "${PRETRAINED_MODEL}/run${SEED}" "${OUTPUT_DIR}/run$SEED"
    fi
    if [ "$file_extension" = "txt" ]; then
        # Read the string from the file
        my_string=$(<"$CONFIG_FILE")

        # Split the string into an array
        read -r -a CONFIG_LIST <<< "$my_string"

        #CONFIG_LIST=$(cat "$CONFIG_FILE") 
        python dreamerv3/train.py "--seed" "$SEED" "--logdir" "${OUTPUT_DIR}/run$SEED" "--configs" "rwrl_vision" "${CONFIG_LIST[@]}"
    else
        python dreamerv3/train.py "--seed" "$SEED" "--logdir" "${OUTPUT_DIR}/run$SEED" "--config" "$CONFIG_FILE"
    fi
done
