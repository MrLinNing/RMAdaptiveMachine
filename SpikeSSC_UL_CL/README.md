# Speaker Recognition Learning, Unlearning & Continual Learning

## Dataset
In speaker recognition tasks, we use the speaker labels of the spiking speech command (SSC) to train the SNN network. The SSC dataset can be downloaded from [here](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/), and the downloaded dataset should be placed in `Dataset/SSC`.

# experiment

### Task1: Initial learning
Run `train_ssc.sh` to train a model learning to recognize 5 speakers.

### Task2: Unlearning with LoRA
Run `train_randlabel_lora.sh` to unlearn the speaker ID1 using the model trained in Task1

### Task3: Continual Learning with LoRA
To learn an additional speaker (ID 6), run the script `train_continual_lora.sh` using the model trained in Task2. In our experiment, we specifically used the model checkpoint from epoch 57 of Task2.

### Task4: Unlearning and Continual Learning by Training All Parameters
Run `train_randlabel_full.sh` and `train_continual_full.sh` in sequence to perform unlearning and continual learning by training all parameters of the backbone network, for comparison with the results of the LoRA method.

### Task5: Simulate the Deployment of Network Weights on Resistive Memory
Run the scripts `rram_eval_lr.sh`, `rram_eval_ul.sh`, and `rram_eval_cl.sh` to evaluate how write noise in resistive memory affects the accuracy of the backbone network trained for learning, unlearning, and continual learning, respectively.

### Task6: Restore the Accuracy with Digital LoRA
Run `train_restore_lora.sh` to train a LoRA for a network whose weights have been disturbed by write noise, thereby restoring its accuracy.