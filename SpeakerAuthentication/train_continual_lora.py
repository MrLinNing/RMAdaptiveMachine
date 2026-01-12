
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from datasets_SSC_speakers import SSC_dataloaders
from models import SNN_LSM_Model

import random
import numpy as np
import argparse
import os

from utils import freeze_all_param, print_trainable_params, labels_to_one_hot, filter_samples, replace_cls_label, random_remove_samples, count_parameters
from lora_patch_multi import patch_model_with_lora, print_model_lora_status

from IncrementalFCReplacer import IncrementalFCReplacer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def arg_parse():
    parser = argparse.ArgumentParser()


    parser.add_argument('--epochs', type=int, default=60,
                        help='num of epochs')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--Tw', type=int, default=10, 
                        help='Time window value of LIF')

    # model infoIncrementalFCReplacer
    parser.add_argument('--mlp', type=str, default="140,512,5",
                        help="model structure of fc layers before continue learning, that is the structure of pretrained model." \
                        " The structure of continue learning model will add one additional node in the last layer for the new class.")
    
    # dataset info
    parser.add_argument('--datasets_path', type=str, default='Datasets/SSC',
                        help="dataset path")
    parser.add_argument('--dataset', type=str, default='SSC',
                        help="dataset type")
    parser.add_argument('--n_bins', type=int, default=5, 
                        help='number of bins for encoding')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size for training')
    parser.add_argument('--num_selected_speakers_continue', type=int, default=7, 
                    help='The number of speakers for continue learning that the dataloader will load their samples from the speakers with the most data.')
    parser.add_argument('--num_selected_speakers_pretrained', type=int, default=5, 
                        help='number of speakers used in pretrained model experiment before continue learning')

    # snn information
    parser.add_argument('--vth', type=float, default=0.3,
                        help='lif neuron vth')
    parser.add_argument('--decay', type=float, default=0.5,
                        help='lif neuron decay')
    parser.add_argument('--const', type=float, default=0.5,
                        help='lif neuron constant')


    ## setup
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=10, help='seed id')
    ## checkname
    parser.add_argument('--name',type=str, default='snn_lsm_512_continue_5cls_10tw')

    # continue learning
    parser.add_argument('--continue_learn_idx', type=int, default=6, help='Index of the class to continue learning')
    parser.add_argument('--previous_unlearn_idx', type=int, default=1, help='Index of the class to unlearn')

    parser.add_argument('--pretrained_model', type=str, default='./checkpoints_randlabel_unlearn/snn_lsm_512_randlabel_5cls_10twat57.pth', help='Path to the pretrained model')
    parser.add_argument('--if_pretrained_lora', type=int, default=1, help='number of LoRA for pretrained model, 0 means no lora applied')

    # lora
    parser.add_argument('--lora_rank_pretrained', type=int, default=8, help='Rank for LoRA in pretrained model')
    parser.add_argument('--lora_alpha_pretrained', type=int, default=16, help='Alpha for LoRA in pretrained model')

    parser.add_argument('--lora_rank_continue', type=int, default=8, help='Rank for LoRA for continue learning')
    parser.add_argument('--lora_alpha_continue', type=int, default=16, help='Alpha for LoRA for continue learning')

    
    return parser.parse_args()

args = arg_parse()


print(args)

setup_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")
cfg_fc = [int(num) for num in args.mlp.split(",")]

assert args.num_selected_speakers_continue > args.num_selected_speakers_pretrained, "num_selected_speakers_continue should be larger than num_selected_speakers_pretrained"
assert args.continue_learn_idx <= args.num_selected_speakers_continue - 1, "continue_learn_idx should be less than num_selected_speakers_continue"
assert args.previous_unlearn_idx <= args.num_selected_speakers_pretrained -1, "previous_unlearn_idx should be less than num_selected_speakers_pretrained"

n_epochs = args.epochs
learning_rate = args.lr
num_selected_speakers_pretrained = args.num_selected_speakers_pretrained
if_pretrained_lora = args.if_pretrained_lora


os.makedirs('checkpoints_continue_learning', exist_ok=True) 
log_file = open(f'checkpoints_continue_learning/{args.name}_log.txt', 'w')

with open(f'checkpoints_continue_learning/{args.name}_config.txt', 'w') as config_file:
    for key, value in vars(args).items():
        config_file.write(f"{key}: {value}\n")



# pretrained model
model = SNN_LSM_Model(cfg=cfg_fc,time_window=args.Tw,thresh=args.vth,decay_lsm=args.decay, const=args.const)
model.to(device)

def custom_filter(name, layer):
    if isinstance(layer, nn.Conv2d):
        """Only patch convolutional layers with output channels greater than 64"""
        return layer.out_channels > 64
    elif isinstance(layer, nn.Linear):
        """Only patch linear layers with large enough parameter count"""
        return layer.in_features * layer.out_features >= 10000
    else:
        return False
if if_pretrained_lora > 0:
    print("The pretrained model has LoRA patches applied.")

    rank = args.lora_rank_pretrained
    lora_alpha = args.lora_alpha_pretrained
    patched_count, total_count, report = patch_model_with_lora(
        model, 
        rank=rank, 
        lora_alpha=lora_alpha,
        target_layers=custom_filter,
        adapter_name="default",
        verbose=True
    )
    print_model_lora_status(model)

    available_class_idx_pretrained = [i for i in range(num_selected_speakers_pretrained) if i != args.previous_unlearn_idx]
    print(f"Available class indices for pretrained model: {available_class_idx_pretrained}")
else:
    print("The pretrained model has NO LoRA patches applied.")
    available_class_idx_pretrained = [i for i in range(num_selected_speakers_pretrained)]
    print(f"Available class indices for pretrained model: {available_class_idx_pretrained}")

networkdata = torch.load(args.pretrained_model)
model.load_state_dict(networkdata)
print("Pretrained model loaded.")

# test pretrained model
train_loader, test_loader = SSC_dataloaders(T = args.Tw, 
                                            num_selected_speakers = args.num_selected_speakers_pretrained, 
                                            root_path = args.datasets_path, 
                                            dataset = args.dataset, 
                                            seed = args.seed, 
                                            batch_size = args.batch_size, 
                                            n_bins=args.n_bins
                                            )
model.eval()
test_correct = 0
test_total = 0
cls_total_dict = {cls_idx: 0 for cls_idx in range(num_selected_speakers_pretrained)}
cls_correct_dict = {cls_idx: 0 for cls_idx in range(num_selected_speakers_pretrained)}
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, label_speakers = data[0], data[2]
        inputs, label_speakers = inputs.to(device), label_speakers.to(device)
        
        # Convert labels to one-hot encoding
        one_hot_labels = labels_to_one_hot(label_speakers, num_classes=cfg_fc[-1])
        
        outputs = model(inputs)

        # functional.reset_net(model) 

        _, predicted = torch.max(outputs, 1)
        test_total += label_speakers.size(0)
        test_correct += predicted.eq(label_speakers).sum().item()

        # acc for each class
        for cls_idx in range(num_selected_speakers_pretrained):
            cls_mask = (label_speakers == cls_idx)
            cls_total = cls_mask.sum().item()
            if cls_total == 0:
                continue
            cls_correct = (predicted[cls_mask] == label_speakers[cls_mask]).sum().item()
            cls_total_dict[cls_idx] += cls_total
            cls_correct_dict[cls_idx] += cls_correct
cls_accs_dict = {}
for cls_idx in range(num_selected_speakers_pretrained):
    if cls_total_dict[cls_idx] == 0:
        continue
    cls_acc = cls_correct_dict[cls_idx] / cls_total_dict[cls_idx]
    cls_accs_dict[cls_idx] = cls_acc
# print acc for each class
for cls_idx, cls_acc in cls_accs_dict.items():
    print(f"Class {cls_idx} accuracy before unlearning: {cls_acc:.4f}")
    with open(f'checkpoints_continue_learning/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Class {cls_idx} accuracy before unlearning: {cls_acc:.4f}\n")

test_acc = test_correct / test_total
print(f"Test accuracy before unlearning: {test_acc:.4f}")
with open(f'checkpoints_continue_learning/{args.name}_log.txt', 'a') as log_file:
    log_file.write(f"Test accuracy before unlearning: {test_acc:.4f}\n")




# start continue learning
#continue learning dataloader setup
train_loader, test_loader = SSC_dataloaders(T = args.Tw, 
                                            num_selected_speakers = args.num_selected_speakers_continue, 
                                            root_path = args.datasets_path, 
                                            dataset = args.dataset, 
                                            seed = args.seed, 
                                            batch_size = args.batch_size, 
                                            n_bins=args.n_bins
                                            )


continue_cls_node_idx = max(available_class_idx_pretrained)+ 1  # index in the continue learning model
num_out_nodes_continue = continue_cls_node_idx + 1

# continue learning model setup
# replace fc classification head for continue learning
replacer = IncrementalFCReplacer(model, 'fc2')
model = replacer.replace_fc_for_incremental_learning(
    new_out_features=num_out_nodes_continue,
    copy_existing_weights=True,
    init_new_weights='xavier',     # new node weights initialization method
    freeze_backbone=True,          # freeze backbone layers
    unfreeze_layers=None           # layers that will not be frozen
)
# read old_fc2 parameters distribution
old_fc2_weight = networkdata['fc2.weight']
old_fc2_bias = networkdata['fc2.bias']
old_weight_mean = old_fc2_weight.mean()
old_weight_std = old_fc2_weight.std()
old_bias_mean = old_fc2_bias.mean()
old_bias_std = old_fc2_bias.std()
# initialize new node weights similar to old weights distribution
with torch.no_grad():
    new_weight = model.fc2.weight[continue_cls_node_idx:, :]
    new_bias = model.fc2.bias[continue_cls_node_idx:]
    new_weight.normal_(mean=old_weight_mean.item(), std=old_weight_std.item())
    new_bias.zero_()
# load new_weight and new_bias into fc2 layer
with torch.no_grad():
    model.fc2.weight[continue_cls_node_idx:, :] = new_weight
    model.fc2.bias[continue_cls_node_idx:] = new_bias

freeze_all_param(model)

def custom_filter(name, layer):
    if isinstance(layer, nn.Conv2d):
        """Only patch convolutional layers with output channels greater than 64"""
        return layer.out_channels > 64
    elif isinstance(layer, nn.Linear):
        """Only patch linear layers with large enough parameter count"""
        return layer.in_features * layer.out_features >= 10000
    else:
        return False

rank = args.lora_rank_continue
lora_alpha = args.lora_alpha_continue
patched_count, total_count, report = patch_model_with_lora(
    model, 
    rank=rank, 
    lora_alpha=lora_alpha,
    target_layers=custom_filter,
    adapter_name="continue_learning",
    verbose=True
)

print_model_lora_status(model)
print_trainable_params(model)

total_params = count_parameters(model)
print(f"After LoRA patching, total model parameters: {total_params}")

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

with open(f'checkpoints_continue_learning/{args.name}_log.txt', 'a') as log_file:
    log_file.write(f"\n-----------------Starting continue learning-----------------------\n")

# continue learning training loop
available_idx_train_continue = available_class_idx_pretrained + [args.continue_learn_idx]
print(f"Available class indices for continue learning model training: {available_idx_train_continue}")
available_idx_test_continue = [i for i in range(args.num_selected_speakers_pretrained)] + [args.continue_learn_idx]
print(f"Available class indices for continue learning model testing: {available_idx_test_continue}")
for epoch in range(n_epochs):

    train_loader.reset()
    # valid_loader.reset()
    test_loader.reset()

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    cls_total_dict_train = {cls_idx: 0 for cls_idx in available_idx_train_continue}
    cls_correct_dict_train = {cls_idx: 0 for cls_idx in available_idx_train_continue}
    continue_cls_acc = 0.0

    loss_lower_bound = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels, label_speakers = data[0], data[1], data[2]
        inputs, label_speakers = inputs.to(device), label_speakers.to(device)

        # filter inputs and labels to only keep available classes
        filtered_inputs, filtered_label_speakers = filter_samples(inputs, label_speakers, available_idx_train_continue)
        # replace continue learning class index to node index in continue learning model
        filtered_label_speakers, mask_continue = replace_cls_label(filtered_label_speakers, old_idx = args.continue_learn_idx, new_idx = continue_cls_node_idx)

        random_inputs, random_labels = random_remove_samples(filtered_inputs, filtered_label_speakers, remove_ratio=0.7, class_idx_list=available_class_idx_pretrained, seed=args.seed, epoch=epoch)  

        one_hot_labels = labels_to_one_hot(random_labels, continue_cls_node_idx + 1)
        
        optimizer.zero_grad()
        outputs = model(random_inputs)

        loss = criterion(outputs, one_hot_labels)
        
        loss.backward()
        optimizer.step()

        # functional.reset_net(model)

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += random_labels.size(0)
        correct += predicted.eq(random_labels).sum().item()

        for cls_idx in available_idx_train_continue:
            if cls_idx == args.continue_learn_idx:
                cls_idx_in_model = continue_cls_node_idx
            else:
                cls_idx_in_model = cls_idx
            cls_mask = (random_labels == cls_idx_in_model)
            cls_total = cls_mask.sum().item()
            if cls_total == 0:
                continue
            cls_correct = (predicted[cls_mask] == random_labels[cls_mask]).sum().item()
            cls_total_dict_train[cls_idx] += cls_total
            cls_correct_dict_train[cls_idx] += cls_correct
    
    cls_accs_dict_train = {}
    for cls_idx in available_idx_train_continue:
        if cls_total_dict_train[cls_idx] == 0:
            continue
        cls_acc = cls_correct_dict_train[cls_idx] / cls_total_dict_train[cls_idx]
        cls_accs_dict_train[cls_idx] = cls_acc
    # print acc for each class
    for cls_idx, cls_acc in cls_accs_dict_train.items():
        print(f"Training Class {cls_idx} accuracy during training: {cls_acc:.4f}")
        with open(f'checkpoints_continue_learning/{args.name}_log.txt', 'a') as log_file:
            log_file.write(f"Training Class {cls_idx} accuracy during training: {cls_acc:.4f}\n")

    # print continue class accuracy and other classes accuracy
    continue_cls_acc_train = cls_accs_dict_train.get(args.continue_learn_idx, None)
    print(f"Training Continue class (index {args.continue_learn_idx}) accuracy during training: {continue_cls_acc_train:.4f}")
    with open(f'checkpoints_continue_learning/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Continue class (index {args.continue_learn_idx}) accuracy during training: {continue_cls_acc_train:.4f}\n")
    other_cls_total_train = sum([total for idx, total in cls_total_dict_train.items() if (idx != args.continue_learn_idx and idx != args.previous_unlearn_idx)])
    other_cls_correct_train = sum([correct for idx, correct in cls_correct_dict_train.items() if (idx != args.continue_learn_idx and idx != args.previous_unlearn_idx)])
    other_cls_acc_train = other_cls_correct_train / other_cls_total_train
    print(f"Training Other classes accuracy during training: {other_cls_acc_train:.4f}")
    with open(f'checkpoints_continue_learning/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Other classes accuracy during training: {other_cls_acc_train:.4f}\n")

    train_loss = total_loss / (i + 1)
    train_acc = correct / total

    


    # Testing loop
    model.eval()
    test_correct = 0
    test_total = 0
    cls_total_dict_test = {cls_idx: 0 for cls_idx in available_idx_test_continue}
    cls_correct_dict_test = {cls_idx: 0 for cls_idx in available_idx_test_continue}

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, label_speakers = data[0], data[2]
            inputs, label_speakers = inputs.to(device), label_speakers.to(device)

            # filter inputs and labels to only keep available classes
            filtered_inputs, filtered_label_speakers = filter_samples(inputs, label_speakers, available_idx_test_continue)
            # replace continue learning class index to node index in continue learning model
            modified_label_speakers, mask_continue = replace_cls_label(filtered_label_speakers, old_idx = args.continue_learn_idx, new_idx = continue_cls_node_idx)

            # Convert labels to one-hot encoding
            one_hot_labels = labels_to_one_hot(modified_label_speakers, continue_cls_node_idx + 1)
            
            outputs = model(filtered_inputs)

            # functional.reset_net(model) 

            _, predicted = torch.max(outputs, 1)
            test_total += modified_label_speakers.size(0)
            test_correct += predicted.eq(modified_label_speakers).sum().item()

            # acc for each class
            for cls_idx in available_idx_test_continue:
                if cls_idx == args.continue_learn_idx:
                    cls_idx_in_model = continue_cls_node_idx
                else:
                    cls_idx_in_model = cls_idx
                cls_mask = (modified_label_speakers == cls_idx_in_model)
                cls_total = cls_mask.sum().item()
                if cls_total == 0:
                    continue
                cls_correct = (predicted[cls_mask] == modified_label_speakers[cls_mask]).sum().item()
                cls_total_dict_test[cls_idx] += cls_total
                cls_correct_dict_test[cls_idx] += cls_correct
    
    cls_accs_dict_test = {}
    for cls_idx in available_idx_test_continue:
        if cls_total_dict_test[cls_idx] == 0:
            continue
        cls_acc = cls_correct_dict_test[cls_idx] / cls_total_dict_test[cls_idx]
        cls_accs_dict_test[cls_idx] = cls_acc
    # print acc for each class
    for cls_idx, cls_acc in cls_accs_dict_test.items():
        print(f"Testing Class {cls_idx} accuracy after continue learning: {cls_acc:.4f}")
        with open(f'checkpoints_continue_learning/{args.name}_log.txt', 'a') as log_file:
            log_file.write(f"Testing Class {cls_idx} accuracy after continue learning: {cls_acc:.4f}\n")
            
    # print continue class accuracy and other classes accuracy
    continue_cls_acc_test = cls_accs_dict_test.get(args.continue_learn_idx, None)
    print(f"Testing Continue class (index {args.continue_learn_idx}) accuracy after continue learning: {continue_cls_acc_test:.4f}")
    with open(f'checkpoints_continue_learning/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Continue class (index {args.continue_learn_idx}) accuracy after continue learning: {continue_cls_acc_test:.4f}\n")
    other_cls_total_test = sum([total for idx, total in cls_total_dict_test.items() if (idx != args.continue_learn_idx and idx != args.previous_unlearn_idx)])
    other_cls_correct_test = sum([correct for idx, correct in cls_correct_dict_test.items() if (idx != args.continue_learn_idx and idx != args.previous_unlearn_idx)])
    other_cls_acc_test = other_cls_correct_test / other_cls_total_test
    print(f"Testing Other classes accuracy after continue learning: {other_cls_acc_test:.4f}")
    with open(f'checkpoints_continue_learning/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Other classes accuracy after continue learning: {other_cls_acc_test:.4f}\n")

    test_acc = test_correct / test_total
        
    print(f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}")
    print('--------------------------------------------\n')
    with open(f'checkpoints_continue_learning/{args.name}_log.txt', 'a') as log_file:
        write_line = f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}\n--------------------------------------------\n"
        log_file.write(write_line)

    if continue_cls_acc_test > 0.90 and other_cls_acc_test > 0.95:
        torch.save(model.state_dict(), f'checkpoints_continue_learning/{args.name}at{epoch+1}.pth')
    
