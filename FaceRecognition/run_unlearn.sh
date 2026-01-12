#!/bin/bash
# Set the log directory path
LOG_DIR="./logs"

# Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR" || {
        echo "Error: Failed to create log directory $LOG_DIR"
        exit 1
    }
fi


python -u main_mixer_unlearn.py --lr 0.01 --unlearn_class 2 --gpu_id 3 --epochs 16  2>&1 | tee -a ./logs/train_unlearning.log
