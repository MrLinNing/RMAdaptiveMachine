import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional, Union

class IncrementalFCReplacer:
    """
    Incremental Learning Fully Connected Layer Replacement Tool
    Used to add new output nodes to an existing model, supporting incremental learning scenarios
    """
    
    def __init__(self, model: nn.Module, fc_layer_name: str = 'fc'):
        """
        Initialize the replacer
        
        Args:
            model: The model to be modified
            fc_layer_name: The attribute name of the fully connected layer, default is 'fc'
        """
        self.model = model
        self.fc_layer_name = fc_layer_name
        self.original_state_dict = model.state_dict().copy()
        self.original_device = next(model.parameters()).device
        
    def get_fc_layer_info(self) -> dict:
        """
        Get information about the current FC layer
        
        Returns:
            dict: A dictionary containing input dimension, output dimension, and other information
        """
        fc_layer = getattr(self.model, self.fc_layer_name)
        return {
            'in_features': fc_layer.in_features,
            'out_features': fc_layer.out_features,
            'weight_shape': fc_layer.weight.shape,
            'bias_shape': fc_layer.bias.shape if fc_layer.bias is not None else None
        }
    
    def replace_fc_for_incremental_learning(
        self, 
        new_out_features: int,
        copy_existing_weights: bool = True,
        init_new_weights: str = 'xavier',
        freeze_backbone: bool = True,
        unfreeze_layers: Optional[list] = None
    ) -> nn.Module:
        """
        Replace the FC layer to support incremental learning
        
        Args:
            new_out_features: The new output dimension (including both old and new classes)
            copy_existing_weights: Whether to copy existing weights
            init_new_weights: Initialization method for new weights ('xavier', 'kaiming', 'normal', 'zeros')
            freeze_backbone: Whether to freeze the backbone network
            unfreeze_layers: List of specific layers to unfreeze
            
        Returns:
            The modified model
        """
        
        # Get current FC layer information
        old_fc = getattr(self.model, self.fc_layer_name)
        old_info = self.get_fc_layer_info()
        
        print(f"Old FC layer: {old_info['in_features']} -> {old_info['out_features']}")
        print(f"New FC layer: {old_info['in_features']} -> {new_out_features}")
        print(f"Added {new_out_features - old_info['out_features']} new output nodes")
        
        # Create new FC layer
        new_fc = nn.Linear(old_info['in_features'], new_out_features).to(self.original_device)
        
        # Copy existing weights and initialize new weights
        if copy_existing_weights:
            with torch.no_grad():
                # Copy old weights to the front part of the new layer
                new_fc.weight.data[:old_info['out_features'], :] = old_fc.weight.data.clone()
                
                # Copy old biases (if any)
                if old_fc.bias is not None:
                    new_fc.bias.data[:old_info['out_features']] = old_fc.bias.data.clone()
                
                # Initialize new weights
                self._initialize_new_weights(
                    new_fc, 
                    old_info['out_features'], 
                    new_out_features, 
                    init_method=init_new_weights
                )
        
        # Replace FC layer
        setattr(self.model, self.fc_layer_name, new_fc)
        
        # Set parameter freezing strategy
        self._set_parameter_freezing(freeze_backbone, unfreeze_layers)
        
        print("FC layer replacement completed!")
        return self.model
    
    def _initialize_new_weights(self, new_fc: nn.Linear, old_dim: int, new_dim: int, 
                               init_method: str = 'xavier'):
        """Initialize weights for new nodes"""
        with torch.no_grad():
            if init_method == 'xavier':
                init.xavier_uniform_(new_fc.weight.data[old_dim:, :])
                if new_fc.bias is not None:
                    init.constant_(new_fc.bias.data[old_dim:], 0.0)
                    
            elif init_method == 'kaiming':
                init.kaiming_uniform_(new_fc.weight.data[old_dim:, :], nonlinearity='relu')
                if new_fc.bias is not None:
                    init.constant_(new_fc.bias.data[old_dim:], 0.0)
                    
            elif init_method == 'normal':
                init.normal_(new_fc.weight.data[old_dim:, :], std=0.02)
                if new_fc.bias is not None:
                    init.normal_(new_fc.bias.data[old_dim:], std=0.02)
                    
            elif init_method == 'zeros':
                init.constant_(new_fc.weight.data[old_dim:, :], 0.0)
                if new_fc.bias is not None:
                    init.constant_(new_fc.bias.data[old_dim:], 0.0)
    
    def _set_parameter_freezing(self, freeze_backbone: bool, unfreeze_layers: Optional[list]):
        """Set parameter freezing strategy"""
        if freeze_backbone:
            # Freeze all parameters
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            
            # Unfreeze FC layer
            for name, param in self.model.named_parameters():
                if self.fc_layer_name in name:
                    param.requires_grad = True
            
            # Unfreeze specific layers
            if unfreeze_layers:
                for layer_name in unfreeze_layers:
                    for name, param in self.model.named_parameters():
                        if layer_name in name:
                            param.requires_grad = True
        
        self._print_trainable_parameters()
    
    def _print_trainable_parameters(self):
        """Print information about trainable parameters"""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable parameters ratio: {trainable_params/total_params*100:.2f}%")
    
    def get_optimizer_config(self, lr: float = 0.001, momentum: float = 0.9):
        """
        Get optimizer configuration suitable for incremental learning
        
        Returns:
            filter function to select parameters that require gradients
        """
        return filter(lambda p: p.requires_grad, self.model.parameters())

# Usage example
def example_usage():
    """
    Usage example: Adding new output nodes to a ResNet model
    """
    import torchvision.models as models
    
    # 1. Load a pretrained model (originally 5 classes)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 5)  # Original 5 classes
    
    # 2. Create replacer
    replacer = IncrementalFCReplacer(model, 'fc')
    
    # 3. Replace FC layer, increase to 6 classes
    new_model = replacer.replace_fc_for_incremental_learning(
        new_out_features=6,           # New number of classes (5+1)
        copy_existing_weights=True,    # Copy existing weights
        init_new_weights='xavier',     # New weights initialization method
        freeze_backbone=True,          # Freeze backbone network
        unfreeze_layers=None           # Do not unfreeze other layers
    )
    
    # 4. Create optimizer suitable for incremental learning
    optimizer = torch.optim.SGD(
        replacer.get_optimizer_config(),
        lr=0.001,
        momentum=0.9
    )
    
    return new_model, optimizer

# Advanced Features: Supports Incremental Learning with Knowledge Distillation
class IncrementalLearningWithDistillation:
    """
    Supports incremental learning with knowledge distillation
    Uses the old model as a teacher to prevent catastrophic forgetting
    """
    
    def __init__(self, old_model: nn.Module, new_model: nn.Module, temperature: float = 3.0):
        self.old_model = old_model
        self.new_model = new_model
        self.temperature = temperature
        self.old_model.eval()  # Set teacher model to evaluation mode
    
    def distillation_loss(self, outputs, labels, old_outputs, alpha: float = 0.5):
        """
        Calculate knowledge distillation loss
        
        Args:
            outputs: Outputs from the new model
            labels: Ground truth labels
            old_outputs: Outputs from the old model
            alpha: Weight for the distillation loss
        """
        # Standard cross-entropy loss
        ce_loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Distillation loss (KL divergence)
        T = self.temperature
        p_old = torch.softmax(old_outputs / T, dim=1)
        p_new = torch.log_softmax(outputs[:, :old_outputs.size(1)] / T, dim=1)
        distill_loss = nn.KLDivLoss()(p_new, p_old) * (T * T)
        
        return alpha * distill_loss + (1 - alpha) * ce_loss

# Test code
if __name__ == "__main__":
    # Run example
    model, optimizer = example_usage()
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of output nodes: {output.shape[1]}")