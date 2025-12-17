
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

from utils import print_trainable_params, labels_to_one_hot

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
                        help='number of time windows')

    # model info
    parser.add_argument('--mlp', type=str, default="140,512,5",
                        help="model structure of fc layers")
    
    # dataset info
    parser.add_argument('--datasets_path', type=str, default='Datasets/SSC',
                        help="dataset path")
    parser.add_argument('--dataset', type=str, default='SSC',
                        help="dataset type")
    parser.add_argument('--n_bins', type=int, default=5, 
                        help='number of bins for encoding')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size for training')
    parser.add_argument('--num_selected_speakers', type=int, default=5, 
                        help='The number of speakers that the dataloader will load their samples from the speakers with the most data.')

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
    parser.add_argument('--name',type=str, default='snn_lsm_512_full_randlabel_5cls_10tw')

    # unlearning
    parser.add_argument('--unlearned_class_idx', type=int, default=1, help='Index of the class to unlearn')
    parser.add_argument('--pretrained_model', type=str, default='./checkpoints/snn_lsm_512_5cls_10tw_best_model.pth', help='Path to the pretrained model')

    
    return parser.parse_args()

args = arg_parse()

# random label for unlearned classes
def random_label(unlearn_idx_list, num_classes,batch_labels):
    available_classes = [i for i in range(num_classes) if i not in unlearn_idx_list]
    device = batch_labels.device
    mask = torch.isin(batch_labels, torch.tensor(unlearn_idx_list, device=device))
    random_indices = torch.randint(low=0, high=len(available_classes), 
                                     size=labels.shape, device=device)
    random_labels = torch.tensor(available_classes, device=device)[random_indices]
    modified_labels = torch.where(mask, random_labels, batch_labels)
    return modified_labels

print(args)

setup_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")
cfg_fc = [int(num) for num in args.mlp.split(",")]



num_selected_speakers = args.num_selected_speakers
unlearned_class_idx = args.unlearned_class_idx

assert cfg_fc[-1] == num_selected_speakers

os.makedirs('checkpoints_randlabel_unlearn_full', exist_ok=True) 
log_file = open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'w')

with open(f'checkpoints_randlabel_unlearn_full/{args.name}_config.txt', 'w') as config_file:
    for key, value in vars(args).items():
        config_file.write(f"{key}: {value}\n")


n_epochs = args.epochs
learning_rate = args.lr


train_loader, test_loader = SSC_dataloaders(T = args.Tw, 
                                            num_selected_speakers = args.num_selected_speakers, 
                                            root_path = args.datasets_path, 
                                            dataset = args.dataset, 
                                            seed = args.seed, 
                                            batch_size = args.batch_size, 
                                            n_bins=args.n_bins
                                            )

model = SNN_LSM_Model(cfg=cfg_fc,time_window=args.Tw,thresh=args.vth,decay_lsm=args.decay, const=args.const)
model.to(device)

networkdata = torch.load(args.pretrained_model)
model.load_state_dict(networkdata)
print("Pretrained model loaded.")

# test before unlearning
model.eval()
test_correct = 0
test_total = 0
cls_total_dict = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}
cls_correct_dict = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, label_speakers = data[0], data[2]
        inputs, label_speakers = inputs.to(device), label_speakers.to(device)
        
        # Convert labels to one-hot encoding
        one_hot_labels = labels_to_one_hot(label_speakers, num_classes=cfg_fc[-1])
        
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        test_total += label_speakers.size(0)
        test_correct += predicted.eq(label_speakers).sum().item()

        # acc for each class
        for cls_idx in range(num_selected_speakers):
            cls_mask = (label_speakers == cls_idx)
            cls_total = cls_mask.sum().item()
            if cls_total == 0:
                continue
            cls_correct = (predicted[cls_mask] == label_speakers[cls_mask]).sum().item()
            cls_total_dict[cls_idx] += cls_total
            cls_correct_dict[cls_idx] += cls_correct
cls_accs_dict = {}
for cls_idx in range(num_selected_speakers):
    if cls_total_dict[cls_idx] == 0:
        continue
    cls_acc = cls_correct_dict[cls_idx] / cls_total_dict[cls_idx]
    cls_accs_dict[cls_idx] = cls_acc
# print acc for each class
for cls_idx, cls_acc in cls_accs_dict.items():
    print(f"Class {cls_idx} accuracy before unlearning: {cls_acc:.4f}")
    with open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Class {cls_idx} accuracy before unlearning: {cls_acc:.4f}\n")

test_acc = test_correct / test_total
print(f"Test accuracy before unlearning: {test_acc:.4f}")
with open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'a') as log_file:
    log_file.write(f"Test accuracy before unlearning: {test_acc:.4f}\n")



print_trainable_params(model)


# Change the criterion to nn.MSELoss()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

with open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'a') as log_file:
    log_file.write(f"\n-----------------Starting unlearning training for class index {unlearned_class_idx}...-----------------------\n")

# unlearning training loop
for epoch in range(n_epochs):

    train_loader.reset()
    # valid_loader.reset()
    test_loader.reset()

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    cls_total_dict_train = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}
    cls_correct_dict_train = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}
    unlearned_cls_acc = 0.0

    loss_lower_bound = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels, label_speakers = data[0], data[1], data[2]
        inputs, label_speakers = inputs.to(device), label_speakers.to(device)

        # Modify labels for unlearned classes
        modified_label_speakers = random_label([unlearned_class_idx], num_selected_speakers, label_speakers)

        one_hot_labels = labels_to_one_hot(modified_label_speakers, num_classes=cfg_fc[-1])
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, one_hot_labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += label_speakers.size(0)
        correct += predicted.eq(label_speakers).sum().item()

        for cls_idx in range(num_selected_speakers):
            cls_mask = (label_speakers == cls_idx)
            cls_total = cls_mask.sum().item()
            if cls_total == 0:
                continue
            cls_correct = (predicted[cls_mask] == label_speakers[cls_mask]).sum().item()
            cls_total_dict_train[cls_idx] += cls_total
            cls_correct_dict_train[cls_idx] += cls_correct
    
    cls_accs_dict_train = {}
    for cls_idx in range(num_selected_speakers):
        if cls_total_dict_train[cls_idx] == 0:
            continue
        cls_acc = cls_correct_dict_train[cls_idx] / cls_total_dict_train[cls_idx]
        cls_accs_dict_train[cls_idx] = cls_acc
    # print acc for each class
    for cls_idx, cls_acc in cls_accs_dict_train.items():
        print(f"Training Class {cls_idx} accuracy during training: {cls_acc:.4f}")
        with open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'a') as log_file:
            log_file.write(f"Training Class {cls_idx} accuracy during training: {cls_acc:.4f}\n")

    # print unlearned class accuracy and other classes accuracy
    unlearned_cls_acc_train = cls_accs_dict_train.get(unlearned_class_idx, None)
    print(f"Training Unlearned class (index {unlearned_class_idx}) accuracy during training: {unlearned_cls_acc_train:.4f}")
    with open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Unlearned class (index {unlearned_class_idx}) accuracy during training: {unlearned_cls_acc_train:.4f}\n")
    other_cls_total_train = sum([total for idx, total in cls_total_dict_train.items() if idx != unlearned_class_idx])
    other_cls_correct_train = sum([correct for idx, correct in cls_correct_dict_train.items() if idx != unlearned_class_idx])
    other_cls_acc_train = other_cls_correct_train / other_cls_total_train
    print(f"Training Other classes accuracy during training: {other_cls_acc_train:.4f}")
    with open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Other classes accuracy during training: {other_cls_acc_train:.4f}\n")

    train_loss = total_loss / (i + 1)
    train_acc = correct / total

    


    # Testing loop
    model.eval()
    test_correct = 0
    test_total = 0
    cls_total_dict_test = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}
    cls_correct_dict_test = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, label_speakers = data[0], data[2]
            inputs, label_speakers = inputs.to(device), label_speakers.to(device)
            
            # Convert labels to one-hot encoding
            one_hot_labels = labels_to_one_hot(label_speakers, num_classes=cfg_fc[-1])
            
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            test_total += label_speakers.size(0)
            test_correct += predicted.eq(label_speakers).sum().item()

            # acc for each class
            for cls_idx in range(num_selected_speakers):
                cls_mask = (label_speakers == cls_idx)
                cls_total = cls_mask.sum().item()
                if cls_total == 0:
                    continue
                cls_correct = (predicted[cls_mask] == label_speakers[cls_mask]).sum().item()
                cls_total_dict_test[cls_idx] += cls_total
                cls_correct_dict_test[cls_idx] += cls_correct
    
    cls_accs_dict_test = {}
    for cls_idx in range(num_selected_speakers):
        if cls_total_dict_test[cls_idx] == 0:
            continue
        cls_acc = cls_correct_dict_test[cls_idx] / cls_total_dict_test[cls_idx]
        cls_accs_dict_test[cls_idx] = cls_acc
    # print acc for each class
    for cls_idx, cls_acc in cls_accs_dict_test.items():
        print(f"Testing Class {cls_idx} accuracy after unlearning: {cls_acc:.4f}")
        with open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'a') as log_file:
            log_file.write(f"Testing Class {cls_idx} accuracy after unlearning: {cls_acc:.4f}\n")
            
    # print unlearned class accuracy and other classes accuracy
    unlearned_cls_acc_test = cls_accs_dict_test.get(unlearned_class_idx, None)
    print(f"Testing Unlearned class (index {unlearned_class_idx}) accuracy after unlearning: {unlearned_cls_acc_test:.4f}")
    with open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Unlearned class (index {unlearned_class_idx}) accuracy after unlearning: {unlearned_cls_acc_test:.4f}\n")
    other_cls_total_test = sum([total for idx, total in cls_total_dict_test.items() if idx != unlearned_class_idx])
    other_cls_correct_test = sum([correct for idx, correct in cls_correct_dict_test.items() if idx != unlearned_class_idx])
    other_cls_acc_test = other_cls_correct_test / other_cls_total_test
    print(f"Testing Other classes accuracy after unlearning: {other_cls_acc_test:.4f}")
    with open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Other classes accuracy after unlearning: {other_cls_acc_test:.4f}\n")
            

    test_acc = test_correct / test_total

        
    print(f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}")
    print('--------------------------------------------\n')
    with open(f'checkpoints_randlabel_unlearn_full/{args.name}_log.txt', 'a') as log_file:
        write_line = f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}\n--------------------------------------------\n"
        log_file.write(write_line)

    if unlearned_cls_acc_test is not None and unlearned_cls_acc_train is not None:
        if unlearned_cls_acc_test < 0.001 and other_cls_acc_test > 0.96:
            print('get a good unlearned result, save the model')
            torch.save(model.state_dict(), f'checkpoints_randlabel_unlearn_full/{args.name}at{epoch+1}.pth')
    
