
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from datasets_SSC_speakers import SSC_dataloaders
from models import  SNN_LSM_Model
import random
import numpy as np
import argparse
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def arg_parse():
    parser = argparse.ArgumentParser()


    parser.add_argument('--epochs', type=int, default=300,
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
    parser.add_argument('--name',type=str, default='snn_lsm_512_5cls_10tw')

    
    return parser.parse_args()

args = arg_parse()

print(args)

setup_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")
cfg_fc = [int(num) for num in args.mlp.split(",")]


# n_bins = 5
# batch_size = args.batch
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

model = SNN_LSM_Model(cfg=cfg_fc, time_window=args.Tw, thresh=args.vth, decay_lsm=args.decay, const=args.const)
model.to(device)


# Change the criterion to nn.MSELoss()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Function to convert labels to one-hot encoding
def labels_to_one_hot(labels, num_classes):
    one_hot_labels = torch.zeros(labels.size(0), num_classes).to(labels.device)
    one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
    return one_hot_labels

os.makedirs('checkpoints', exist_ok=True) 
log_file = open(f'checkpoints/{args.name}_log.txt', 'w')
# Training loop
best_test_acc = 0
# best_valid_acc = 0

for epoch in range(n_epochs):

    train_loader.reset()
    # valid_loader.reset()
    test_loader.reset()

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, data in enumerate(train_loader):
        inputs, labels, label_speakers = data[0], data[1], data[2]
        inputs, label_speakers = inputs.to(device), label_speakers.to(device)
        
        # Convert labels to one-hot encoding
        one_hot_labels = labels_to_one_hot(label_speakers, num_classes=cfg_fc[-1])
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, one_hot_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += label_speakers.size(0)
        correct += predicted.eq(label_speakers).sum().item()

    train_loss = total_loss / (i + 1)
    train_acc = correct / total


    # Testing loop
    model.eval()
    test_correct = 0
    test_total = 0

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

        test_acc = test_correct / test_total

    if test_acc > best_test_acc:
        best_test_acc = test_acc

        torch.save(model.state_dict(), f'checkpoints/{args.name}_best_model.pth')
        print(f"new best model saved: {test_acc:.4f}")

    # if valid_acc > best_valid_acc:
    #     best_valid_acc = valid_acc
    with open(f'checkpoints/{args.name}_log.txt', 'a') as log_file:
        write_line = f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Best Test Acc: {best_test_acc:.4f}\n"
        log_file.write(write_line)
        
    print(f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Best Test Acc: {best_test_acc:.4f}")
