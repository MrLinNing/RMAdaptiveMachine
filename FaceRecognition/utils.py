import torch
import os
from torch import nn
from typing import Callable, Optional, Tuple, Union
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict, Any, AnyStr
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

def plot_gallery(title, images, n_col: Optional[int] = 3, n_row: Optional[int] = 2, cmap=plt.cm.gray):
    """Plot images in a gallery format"""
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape((64, 64)),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()


def binary_crossentropy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
    ) -> torch.Tensor:
    """Compute binary cross-entropy loss"""
    return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))


def multiclass_crossentropy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
    ) -> torch.Tensor:
    """Compute multiclass cross-entropy loss"""
    return -torch.mean(torch.sum(y_true * torch.log(y_pred), dim=1))


class AdamOptimizer():
    """Custom Adam optimizer implementation"""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        """Perform one optimization step"""
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        """Zero out all gradients"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


def plot_tsne(model, features, labels, num_classes, device, title='t-SNE Visualization', save_dir="./figure_plot/learn_results", filename="tsne.pdf"):
    """
    Plot t-SNE visualization of test data embeddings using the trained model
    
    Parameters:
        model: Trained best model
        features: Test data features (torch.Tensor)
        labels: Test data true labels (torch.Tensor)
        num_classes: Number of classes (int)
        device: Computation device (torch.device)
        title: Plot title (str)
        save_dir: Directory to save the PDF file (str)
        filename: Name of the saved PDF file (str)
    """
    # Ensure model and data are on the correct device
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    
    # Get feature representations (last hidden layer output)
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'get_features'):  # If model has feature extraction method
            embeddings = model.get_features(features)
        else:  # Default: take the last layer output
            embeddings = model(features)
    
    # Convert to numpy
    X = embeddings.cpu().numpy()
    y = labels.cpu().numpy()
    
    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, 
                perplexity=30, 
                random_state=42,
                n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # Create color palette (Nature-style low-saturation colors)
    palette = sns.color_palette("husl", num_classes)
    
    # Plot t-SNE
    plt.figure(figsize=(5, 4))
    scatter = sns.scatterplot(
        x=X_tsne[:, 0], 
        y=X_tsne[:, 1],
        hue=y,
        palette=palette,
        s=150,
        alpha=0.9,
        edgecolor='black',
        linewidth=0.3
    )
    
    # Add legend and title
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    legend = plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure if path is provided
    if save_dir is not None and filename is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()


def plot_confusion_matrix(model, features, labels, num_classes, device, normalize=True, title='Confusion Matrix', save_path="./figure_plot/learn_results/confusion_matrix.pdf", exclude_label=5):
    """
    Plot confusion matrix for model predictions
    
    Parameters:
        model: Trained PyTorch model
        features: Test features (torch.Tensor)
        labels: True labels (torch.Tensor)
        num_classes: Number of classes (int)
        device: Device for computation (torch.device)
        normalize: Whether to normalize the confusion matrix (bool)
        title: Plot title (str)
        save_path: Path to save PDF file (str), if None then don't save
        exclude_label: Label to exclude (int), if None then no exclusion
    """
    # Move model and data to correct device
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        _, preds = torch.max(outputs, 1)
    
    # Convert to numpy
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    
    # Exclude specific label if requested
    if exclude_label is not None:
        keep_indices = [i for i in range(num_classes) if i != exclude_label]
        cm = cm[keep_indices][:, keep_indices]
        num_classes -= 1
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create class labels excluding the excluded one if needed
    class_labels = [str(i) for i in range(num_classes + 1) if i != exclude_label] if exclude_label is not None else np.arange(num_classes)

    # Light blue colormap
    light_blue = LinearSegmentedColormap.from_list("light_blue", ["#E6F0FF", "#5E9EFF"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=light_blue, 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save as PDF
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.show()
    plt.close()


def calculate_class_accuracy(model: nn.Module, 
                            test_features: torch.Tensor, 
                            test_labels: torch.Tensor, 
                            num_classes: int,
                            device: torch.device):
    """Calculate and print per-class accuracy on test set"""
    model.eval()
    class_correct = torch.zeros(num_classes, dtype=torch.float).to(device)
    class_total = torch.zeros(num_classes, dtype=torch.float).to(device)
    
    with torch.no_grad():
        outputs = model(test_features.to(device))
        _, predicted = torch.max(outputs, 1)
        
        for c in range(num_classes):
            mask = (test_labels.to(device) == c)
            if mask.any():
                class_correct[c] = (predicted[mask] == test_labels.to(device)[mask]).sum()
                class_total[c] = mask.sum()
    
    # Print per-class accuracy
    for c in range(num_classes):
        if class_total[c] > 0:
            acc = class_correct[c].item() / class_total[c].item()
            print(f'Class {c}: {acc:.4f} ({int(class_correct[c])}/{int(class_total[c])})')
        else:
            print(f'Class {c}: N/A (no samples)')


def train(
    training_features: torch.Tensor,
    training_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    epochs: int,
    args,
    device: torch.device):
    """
    Standard training loop with loss/accuracy tracking and best model saving
    """
    # Move data to device
    training_features = training_features.to(device)
    training_labels = training_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    model = model.to(device)
    
    training_loss: List[float] = []
    test_loss: List[float] = []
    training_accuracy: List[float] = []
    test_accuracy: List[float] = []
    best_model_state = None

    num_classes = 6
    best_test_acc = 0.0

    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

    for epoch in range(epochs):
        model.train()

        training_ypred = model(training_features)
        train_loss = loss_function(training_ypred, training_labels)
        training_loss.append(train_loss.item())

        _, predicted = torch.max(training_ypred, 1)
        correct = (predicted == training_labels).sum().item()
        accuracy = correct / training_labels.size(0)
        training_accuracy.append(accuracy)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_ypred = model(test_features)
            val_loss = loss_function(test_ypred, test_labels)
            test_loss.append(val_loss.item())

            _, val_predicted = torch.max(test_ypred, 1)
            val_correct = (val_predicted == test_labels).sum().item()
            val_accuracy = val_correct / test_labels.size(0)
            test_accuracy.append(val_accuracy)

        if val_accuracy > best_test_acc:
            best_test_acc = val_accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"checkpoint/best_{args.model_name}_model.pth")

        print(f"Epoch {epoch}: Training Loss = {train_loss.item():.4f} | Test Loss = {val_loss.item():.4f} | Training Acc = {accuracy:.4f} | Test Acc = {val_accuracy:.4f} | Best Test Acc = {best_test_acc:.4f}")

    # Load best model and evaluate per-class accuracy
    model.load_state_dict(best_model_state)
    model.eval()
    calculate_class_accuracy(model, test_features, test_labels, num_classes, device)

    print("Generating t-SNE visualization for test data:")
    plot_tsne(model, test_features, test_labels, num_classes, device, 
              title=f't-SNE of {args.model_name}', save_dir=f"./figure_plot/{args.model_name}", filename=f"tsne.pdf")

    return training_loss, test_loss, training_accuracy, test_accuracy


def custom_loss(output, target, class_weights):
    """Weighted cross-entropy loss"""
    loss = torch.nn.CrossEntropyLoss(reduction='none')(output, target)
    weighted_loss = torch.mul(loss, class_weights[target])
    return torch.mean(weighted_loss)


def train_with_gradient_ascent(
    training_features: torch.Tensor,
    training_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    epochs: int,
    target_classes: List[int], 
    class_weight_values: List[float],
    device: torch.device,
    args):
    """
    Training loop with class-weighted loss (gradient ascent style for specific classes)
    """
    training_loss: List[float] = []
    test_loss: List[float] = []
    training_accuracy: List[float] = []
    overall_test_accuracy: List[float] = []
    target_test_accuracy: List[float] = []
    other_test_accuracy: List[float] = []

    num_classes = 6

    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

    class_weights = torch.ones(num_classes)
    for target_class in target_classes:
        class_weights[target_class] = class_weight_values
    class_weights = class_weights.to(device)

    for epoch in range(epochs):
        model.train()

        training_ypred = model(training_features.to(device))
        train_loss = custom_loss(training_ypred, training_labels.to(device), class_weights)
        training_loss.append(train_loss.item())

        _, predicted = torch.max(training_ypred, 1)
        correct = (predicted == training_labels.to(device)).sum().item()
        accuracy = correct / training_labels.size(0)
        training_accuracy.append(accuracy)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_ypred = model(test_features.to(device))
            val_loss = loss_function(test_ypred, test_labels.to(device))
            test_loss.append(val_loss.item())

            _, val_predicted = torch.max(test_ypred, 1)
            val_correct = (val_predicted == test_labels.to(device)).sum().item()
            overall_acc = val_correct / test_labels.size(0)
            overall_test_accuracy.append(overall_acc)
            
            # Target classes accuracy
            target_mask = torch.isin(test_labels.to(device), torch.tensor(target_classes, device=device))
            target_acc = 0.0
            if target_mask.any():
                target_preds = val_predicted[target_mask]
                target_labels = test_labels.to(device)[target_mask]
                target_correct = (target_preds == target_labels).sum().item()
                target_acc = target_correct / target_mask.sum().item()
            target_test_accuracy.append(target_acc)
            
            # Other classes accuracy
            other_mask = ~target_mask
            other_acc = 0.0
            if other_mask.any():
                other_preds = val_predicted[other_mask]
                other_labels = test_labels.to(device)[other_mask]
                other_correct = (other_preds == other_labels).sum().item()
                other_acc = other_correct / other_mask.sum().item()
            other_test_accuracy.append(other_acc)

        print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f} | Test Loss = {val_loss.item():.4f}")
        print(f"  Train Acc = {accuracy:.4f} | Overall Test Acc = {overall_acc:.4f}")
        print(f"  Target Classes ({target_classes}) Test Acc = {target_acc:.4f}")
        print(f"  Other Classes Test Acc = {other_acc:.4f}")

        print("\nPer-class Test Accuracy:")
        calculate_class_accuracy(model, test_features, test_labels, num_classes, device)
        print("="*50 + "\n")

    best_model_state = model.state_dict()
    torch.save(best_model_state, f"checkpoint/best_{args.model_name}_model.pth")

    print("Generating t-SNE visualization for test data:")
    plot_tsne(model, test_features, test_labels, num_classes, device, 
              title=f't-SNE of {args.model_name}', save_dir=f"./figure_plot/{args.model_name}", filename=f"tsne.pdf")

    return training_loss, test_loss, training_accuracy, overall_test_accuracy, target_test_accuracy, other_test_accuracy


def train_from_scratch(
    training_features: torch.Tensor,
    training_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    epochs: int,
    args,
    device: torch.device):
    """
    Training from scratch with special tracking for unlearning target class(es)
    """
    training_features = training_features.to(device)
    training_labels = training_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    model = model.to(device)
    
    training_loss: List[float] = []
    test_loss: List[float] = []
    training_accuracy: List[float] = []
    overall_test_accuracy: List[float] = []
    target_test_accuracy: List[float] = []
    other_test_accuracy: List[float] = []
    best_model_state = None

    num_classes = 5

    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

    for epoch in range(epochs):
        model.train()

        training_ypred = model(training_features)
        train_loss = loss_function(training_ypred, training_labels)
        training_loss.append(train_loss.item())

        _, predicted = torch.max(training_ypred, 1)
        correct = (predicted == training_labels).sum().item()
        accuracy = correct / training_labels.size(0)
        training_accuracy.append(accuracy)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_ypred = model(test_features)
            val_loss = loss_function(test_ypred, test_labels)
            test_loss.append(val_loss.item())

            _, val_predicted = torch.max(test_ypred, 1)
            val_correct = (val_predicted == test_labels).sum().item()
            overall_acc = val_correct / test_labels.size(0)
            overall_test_accuracy.append(overall_acc)
            
            target_mask = torch.isin(test_labels, torch.tensor(args.unlearn_class, device=device))
            target_acc = 0.0
            if target_mask.any():
                target_preds = val_predicted[target_mask]
                target_labels = test_labels[target_mask]
                target_correct = (target_preds == target_labels).sum().item()
                target_acc = target_correct / target_mask.sum().item()
            target_test_accuracy.append(target_acc)
            
            other_mask = ~target_mask
            other_acc = 0.0
            if other_mask.any():
                other_preds = val_predicted[other_mask]
                other_labels = test_labels[other_mask]
                other_correct = (other_preds == other_labels).sum().item()
                other_acc = other_correct / other_mask.sum().item()
            other_test_accuracy.append(other_acc)

        print(f"Epoch {epoch}: Training Loss = {train_loss.item():.4f} | Test Loss = {val_loss.item():.4f} | Training Acc = {accuracy:.4f} | Test Acc = {overall_acc:.4f} | Target Test Acc = {target_acc:.4f} | Remain Test Acc = {other_acc:.4f}")

    best_model_state = model.state_dict()
    torch.save(best_model_state, f"checkpoint/model_{args.model_name}.pth")

    print("Calculating per-class accuracy for the best model:")
    calculate_class_accuracy(model, test_features, test_labels, num_classes, device)

    return training_loss, test_loss, training_accuracy, overall_test_accuracy, target_test_accuracy, other_test_accuracy


def test(
    model: nn.Module,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    model_path: str,
    args=None
):
    """
    Load pretrained model and evaluate performance on test set
    
    Parameters:
        model: Model architecture
        test_features: Test features (N, C, H, W)
        test_labels: Test labels (N,)
        num_classes: Number of classes
        device: Computation device
        model_path: Path to pretrained model
        args: Optional argument object
    """
    # Load pretrained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    # Overall accuracy
    with torch.no_grad():
        outputs = model(test_features)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == test_labels).sum().item()
        accuracy = correct / test_labels.size(0)
    
    print(f"Loaded model from {model_path}")
    print(f"Overall Test Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    with torch.no_grad():
        outputs = model(test_features)
        _, predictions = torch.max(outputs, 1)
        
        for label in range(num_classes):
            mask = (test_labels == label)
            class_total[label] = mask.sum().item()
            class_correct[label] = (predictions[mask] == label).sum().item()
    
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"Class {i}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    # Confusion matrix (in percentages)
    print("\nConfusion Matrix (Percentages):")
    conf_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for i in range(num_classes):
            for j in range(num_classes):
                conf_matrix[i, j] = ((test_labels == i) & (predictions == j)).sum().item()

    row_sums = conf_matrix.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1
    conf_matrix_percent = (conf_matrix / row_sums) * 100

    print(f"True\\Pred", end="")
    for j in range(num_classes):
        print(f"{j:>10}", end="")
    print()
    for i in range(num_classes):
        print(f"{i:>10}", end="")
        for j in range(num_classes):
            print(f"{conf_matrix_percent[i, j]:>10.2f}%", end="")
        print()
    
    # t-SNE visualization
    print("\nGenerating t-SNE visualization:")
    plot_tsne(model, test_features, test_labels, num_classes, device,
              title=f't-SNE (Test Acc={accuracy:.2f})')