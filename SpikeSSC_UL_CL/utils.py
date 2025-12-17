import numpy as np
import random, sys
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_versions():
    python_version = sys.version .split(' ')[0]
    print("============== Checking Packages versions ================")
    print(f"python {python_version}")
    print(f"numpy {np.__version__}")
    print(f"pytorch {torch.__version__}")



def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # This flag only allows cudnn algorithms that are determinestic unlike .benchmark
    torch.backends.cudnn.deterministic = True

    #this flag enables cudnn for some operations such as conv layers and RNNs, 
    # which can yield a significant speedup.
    torch.backends.cudnn.enabled = False

    # This flag enables the cudnn auto-tuner that finds the best algorithm to use
    # for a particular configuration. (this mode is good whenever input sizes do not vary)
    torch.backends.cudnn.benchmark = False

    # I don't know if this is useful, look it up.
    #os.environ['PYTHONHASHSEED'] = str(seed)

def freeze_all_param(model):
    for param in model.parameters():
        param.requires_grad = False

def print_trainable_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name} (requires_grad: {param.requires_grad})")

# Function to convert labels to one-hot encoding
def labels_to_one_hot(labels, num_classes):
    one_hot_labels = torch.zeros(labels.size(0), num_classes).to(labels.device)
    one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
    return one_hot_labels

def filter_samples(inputs, labels, class_list):
    """
    Filter TensorDataset type data
    :param features: feature tensor
    :param labels: label tensor
    :param unlearn_idx_list: list of categories to be removed
    :return: filtered features and labels
    """
    # Create a boolean mask to identify samples to be retained
    device = inputs.device
    class_tensor = torch.tensor(class_list, device=device)
    mask = torch.isin(labels, class_tensor)
    
    # Use boolean indexing to filter data
    filtered_features = inputs[mask]
    filtered_labels = labels[mask]

    
    return filtered_features, filtered_labels

def filter_out_samples(inputs, labels, unlearn_idx_list):
    """
    Filter TensorDataset type data
    :param features: Feature tensor
    :param labels: Label tensor
    :param unlearn_idx_list: List of categories to be removed
    :return: Filtered features and labels
    """
    # Create a boolean mask to identify samples to be retained
    device = inputs.device
    unlearn_tensor = torch.tensor(unlearn_idx_list, device=device)
    mask = ~torch.isin(labels, unlearn_tensor)
    
    # Use boolean indexing to filter data
    filtered_features = inputs[mask]
    filtered_labels = labels[mask]
    
    
    return filtered_features, filtered_labels

def replace_cls_label(labels, old_idx, new_idx):
    """
    Replace old_idx with new_idx in the labels
    :param labels: Label tensor
    :param old_idx: Old category index
    :param new_idx: New category index
    :return: The label tensor after replacement
    """
    mask = labels == old_idx
    labels[mask] = new_idx
    return labels, mask

def random_remove_samples(inputs, labels, remove_ratio, class_idx_list, seed=42, epoch=0):
    """
    Randomly remove a portion of samples from the specified category
    :param features: feature tensor
    :param labels: label tensor
    :param remove_ratio: removal ratio (between 0 and 1)
    :param class_idx: specified category index
    :return: features and labels after removal
    """
    # Create a boolean mask to identify samples to be retained
    device = inputs.device
    for class_idx in class_idx_list:
        class_mask = labels == class_idx
        class_indices = torch.nonzero(class_mask).squeeze()

        if class_indices.numel() == 0 or class_indices.numel() == 1:
            continue

        # Set random seed for reproducibility
        rng = np.random.default_rng(seed + epoch)  # Use a different seed for each epoch

        # Calculate the number of samples to remove
        num_samples_to_remove = int(len(class_indices) * remove_ratio)

        # Randomly select indices of samples to remove
        if num_samples_to_remove > 0:
            remove_indices = rng.choice(class_indices.cpu().numpy(), size=num_samples_to_remove, replace=False)

            # Create a mask of all True, then set positions of samples to be removed to False
            keep_mask = torch.ones(len(labels), dtype=torch.bool, device=device)
            keep_mask[remove_indices] = False

            # Use boolean indexing to filter data
            inputs = inputs[keep_mask]
            labels = labels[keep_mask]
    
    return inputs, labels


