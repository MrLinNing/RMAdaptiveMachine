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


# python -u main_mixer_baseline_continual_learn.py --lr 0.05 --unlearn_class 0 --epochs 200  2>&1 | tee -a ./logs/train_continual_learning_baseline.log
# python -u main_mixer_baseline_continual_learn.py --lr 0.05 --unlearn_class 1 --epochs 200  2>&1 | tee -a ./logs/train_continual_learning_baseline.log
python -u main_mixer_baseline_continual_learn.py --lr 0.005 --unlearn_class 2 --epochs 200  2>&1 | tee -a ./logs/train_continual_learning_baseline.log
# python -u main_mixer_baseline_continual_learn.py --lr 0.05 --unlearn_class 3 --epochs 200  2>&1 | tee -a ./logs/train_continual_learning_baseline.log
# python -u main_mixer_baseline_continual_learn.py --lr 0.05 --unlearn_class 4 --epochs 200  2>&1 | tee -a ./logs/train_continual_learning_baseline.log

