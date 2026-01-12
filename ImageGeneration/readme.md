# Canvas10 Dataset and Model Training Guide

## Overview
This document provides instructions for dataset setup, model training, and inference procedures for the Canvas10 project.

## Dataset Setup
1. Download the dataset and place it in the path: `dataset/canvas10`

## Training Procedures

### Task 1: Initial Training
- **Training script**: `train_task1.sh`
- **Output directory**: Trained models will be saved in `model/task1/`

### Task 2: Unlearning
- **Training script**: `train_task2.sh`
- **Key parameters**:
  - Set `--ckpt-path` parameter to the corresponding model path from Task 1
- **Output directory**: Trained LoRA models will be saved in `model/task2/`

### Task 3: Continual Learning
- **Training script**: `train_task3.sh`
- **Key parameters**:
  - Set `--ckpt-path` parameter to the corresponding model path from Task 1
  - Set `--ckpt-lora-path` parameter to the corresponding model path from Task 2
- **Output directory**: Trained LoRA models will be saved in `model/task3/`

## Inference
1. Open the `sample.ipynb` file
2. In the first code block, modify the following path parameters to point to your trained models:

```python
"--ckpt", "model/task1/0100000.pt",
# "--lora-ckpt", "model/task2/0100000.pt",
# "--lora-ckpt-cl", "model/task3/0100000.pt",
```

3. Run the notebook cells sequentially

### Inference Scenarios
- **Task 1 results**: Use only the `--ckpt` parameter
- **Task 2 results**: Use both `--ckpt` and `--lora-ckpt` parameters
- **Task 3 results**: Use all three parameters (`--ckpt`, `--lora-ckpt`, and `--lora-ckpt-cl`)
