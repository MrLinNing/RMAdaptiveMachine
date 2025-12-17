import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Union, Callable

class LoRALayer(nn.Module):
    """LoRA Base Layer Class"""
    def __init__(self, in_dim, out_dim, rank=8, scaling=1.0):
        super().__init__()
        self.rank = rank
        self.scaling = scaling
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.empty(rank, out_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
    
    def forward(self):
        """Return the product of low-rank matrices"""
        return self.A @ self.B

def patch_conv2d_with_lora(conv_layer, rank=8, lora_alpha=1.0, lora_dropout=0.0, adapter_name="default"):
    """
    Add a LoRA adapter to a single convolutional layer, supporting multiple adapters.

        Args:

        conv_layer: The original convolutional layer to be modified

        rank: The rank of LoRA

        lora_alpha: The alpha parameter of LoRA

        lora_dropout: The dropout rate of LoRA

        adapter_name: The name of the adapter, used to support multiple adapters
    """
    # Initialize adapter storage structure
    if not hasattr(conv_layer, '_lora_adapters'):
        conv_layer._lora_adapters = {}
    
    # Check if an adapter with the same name already exists
    if adapter_name in conv_layer._lora_adapters:
        print(f"Adapter '{adapter_name}' already exists, skipping creation")
        return conv_layer, False
    
    # Get the device of the original weights
    original_device = next(conv_layer.parameters()).device
    
    # Mark as patched
    conv_layer._is_lora_patched = True
    
    # Save original forward method (only save on first patch)
    if not hasattr(conv_layer, '_orig_forward'):
        conv_layer._orig_forward = conv_layer.forward
    
    # Get convolutional layer parameters
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size[0] if isinstance(conv_layer.kernel_size, tuple) else conv_layer.kernel_size
    
    # Create adapter configuration
    adapter_config = {
        'rank': rank,
        'lora_alpha': lora_alpha,
        'lora_scaling': lora_alpha / rank,
        'adapter_name': adapter_name
    }
    
    # For the first patch, use a compatibility naming convention; for subsequent patches, use a naming convention that includes the adapter name.
    if len(conv_layer._lora_adapters) == 0:
        # First patch, use compatibility naming
        lora_A_name = 'lora_A'
        lora_B_name = 'lora_B'
        dropout_name = 'lora_dropout'
    else:
        # Subsequent patches, use naming that includes the adapter name
        lora_A_name = f'lora_A_{adapter_name}'
        lora_B_name = f'lora_B_{adapter_name}'
        dropout_name = f'lora_dropout_{adapter_name}'
    
    # Create LoRA parameters
    setattr(conv_layer, lora_A_name, nn.Parameter(
        torch.zeros(rank * kernel_size, in_channels * kernel_size, device=original_device)
    ))
    setattr(conv_layer, lora_B_name, nn.Parameter(
        torch.zeros(out_channels * kernel_size, rank * kernel_size, device=original_device)
    ))
    
    # Dropout layer
    if lora_dropout > 0:
        setattr(conv_layer, dropout_name, nn.Dropout(lora_dropout).to(original_device))
    else:
        setattr(conv_layer, dropout_name, lambda x: x)
    
    # Initialize LoRA parameters
    nn.init.kaiming_uniform_(getattr(conv_layer, lora_A_name), a=math.sqrt(5))
    nn.init.zeros_(getattr(conv_layer, lora_B_name))
    
    # Freeze original weights only on the first patch
    if len(conv_layer._lora_adapters) == 0:
        conv_layer.weight.requires_grad = False
        if conv_layer.bias is not None:
            conv_layer.bias.requires_grad = False
    
    # Store adapter configuration
    adapter_config.update({
        'lora_A_name': lora_A_name,
        'lora_B_name': lora_B_name,
        'dropout_name': dropout_name
    })
    conv_layer._lora_adapters[adapter_name] = adapter_config
    
    def lora_forward(self, x):
        # Calculate the sum of LoRA weight updates for all adapters
        total_lora_weight = 0
        
        for adapter_name, config in self._lora_adapters.items():
            lora_A = getattr(self, config['lora_A_name'])
            lora_B = getattr(self, config['lora_B_name'])
            dropout = getattr(self, config['dropout_name'])
            
            # Calculate the LoRA weight for the current adapter
            lora_weight = (lora_B @ lora_A).view(self.weight.shape) * config['lora_scaling']
            total_lora_weight += lora_weight
            
            # Apply dropout only for the first adapter (to avoid repeated application)
            if adapter_name == list(self._lora_adapters.keys())[0] and callable(dropout):
                x = dropout(x)
        
        new_weight = self.weight + total_lora_weight
        
        # Convolution with new weights
        return F.conv2d(
            x, new_weight, self.bias, 
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
    
    # Replace forward method
    conv_layer.forward = lora_forward.__get__(conv_layer)
    
    # Set LoRA parameters as trainable
    for param_name in [lora_A_name, lora_B_name]:
        param = getattr(conv_layer, param_name)
        param.requires_grad = True
    
    return conv_layer, True

def patch_linear_with_lora(linear_layer, rank=8, lora_alpha=1.0, lora_dropout=0.0, adapter_name="default"):
    """
    Add a LoRA adapter to a single linear layer, supporting multiple adapters
    
    Args:
        linear_layer: The original linear layer to be modified
        rank: LoRA's rank
        lora_alpha: LoRA's alpha parameter
        lora_dropout: LoRA's dropout rate
        adapter_name: Adapter name for supporting multiple adapters
    """
    # Initialize adapter storage structure
    if not hasattr(linear_layer, '_lora_adapters'):
        linear_layer._lora_adapters = {}
    
    # Check if an adapter with the same name already exists
    if adapter_name in linear_layer._lora_adapters:
        print(f"Adapter '{adapter_name}' already exists, skipping creation")
        return linear_layer, False
    
    # Get the device of the original weights
    original_device = next(linear_layer.parameters()).device
    
    # Mark as patched
    linear_layer._is_lora_patched = True
    
    # Save original forward method (only on the first patch)
    if not hasattr(linear_layer, '_orig_forward'):
        linear_layer._orig_forward = linear_layer.forward
    
    # Get linear layer parameters
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    
    # Create adapter configuration
    adapter_config = {
        'rank': rank,
        'lora_alpha': lora_alpha,
        'lora_scaling': lora_alpha / rank,
        'adapter_name': adapter_name
    }
    
    # Use compatibility naming for the first patch, and adapter name-based naming for subsequent patches
    if len(linear_layer._lora_adapters) == 0:
        # First patch, use compatibility naming
        lora_A_name = 'lora_A'
        lora_B_name = 'lora_B'
        dropout_name = 'lora_dropout'
    else:
        # Subsequent patches, use adapter name-based naming
        lora_A_name = f'lora_A_{adapter_name}'
        lora_B_name = f'lora_B_{adapter_name}'
        dropout_name = f'lora_dropout_{adapter_name}'
    
    # Create LoRA parameters
    setattr(linear_layer, lora_A_name, nn.Parameter(
        torch.zeros(rank, in_features, device=original_device)
    ))
    setattr(linear_layer, lora_B_name, nn.Parameter(
        torch.zeros(out_features, rank, device=original_device)
    ))
    
    # Dropout layer
    if lora_dropout > 0:
        setattr(linear_layer, dropout_name, nn.Dropout(lora_dropout).to(original_device))
    else:
        setattr(linear_layer, dropout_name, lambda x: x)
    
    # Initialize LoRA parameters
    nn.init.kaiming_uniform_(getattr(linear_layer, lora_A_name), a=math.sqrt(5))
    nn.init.zeros_(getattr(linear_layer, lora_B_name))
    
    # Freeze original weights only on the first patch
    if len(linear_layer._lora_adapters) == 0:
        linear_layer.weight.requires_grad = False
        if linear_layer.bias is not None:
            linear_layer.bias.requires_grad = False
    
    # Store adapter configuration
    adapter_config.update({
        'lora_A_name': lora_A_name,
        'lora_B_name': lora_B_name,
        'dropout_name': dropout_name
    })
    linear_layer._lora_adapters[adapter_name] = adapter_config
    
    def lora_forward(self, x):
        # Calculate the sum of LoRA weight updates from all adapters
        total_lora_weight = 0
        
        for adapter_name, config in self._lora_adapters.items():
            lora_A = getattr(self, config['lora_A_name'])
            lora_B = getattr(self, config['lora_B_name'])
            dropout = getattr(self, config['dropout_name'])
            
            # Calculate the LoRA weight for the current adapter
            lora_weight = (lora_B @ lora_A) * config['lora_scaling']
            total_lora_weight += lora_weight
            
            # Apply dropout only for the first adapter (to avoid repeated application)
            if adapter_name == list(self._lora_adapters.keys())[0] and callable(dropout):
                x = dropout(x)
        
        new_weight = self.weight + total_lora_weight
        
        # Use the new weight for the linear transformation
        return F.linear(x, new_weight, self.bias)
    
    # Replace the forward method
    linear_layer.forward = lora_forward.__get__(linear_layer)
    
    # Set LoRA parameters as trainable
    for param_name in [lora_A_name, lora_B_name]:
        param = getattr(linear_layer, param_name)
        param.requires_grad = True
    
    return linear_layer, True

def patch_model_with_lora(
    model: nn.Module, 
    rank: int = 8, 
    lora_alpha: float = 1.0, 
    lora_dropout: float = 0.0,
    target_layers: Union[str, List[str], Callable] = None,
    layer_types: List[str] = None,
    adapter_name: str = "default",  # New: adapter name
    verbose: bool = True
):
    """
    Add LoRA adapters to convolutional and fully connected layers in the entire model, supporting multiple adapters
    
    Args:
        model: The PyTorch model to be modified
        rank: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_layers: Union[str, List[str], Callable] = None,
        layer_types: List[str] = None,
        adapter_name: str = "default",  # New: adapter name
        verbose: bool = True
    """
    
    if layer_types is None:
        layer_types = ['Conv2d', 'Linear']
    
    def should_patch_layer(name, layer):
        """Determine whether to patch this layer"""
        if isinstance(layer, nn.Conv2d) and 'Conv2d' not in layer_types:
            return False
        if isinstance(layer, nn.Linear) and 'Linear' not in layer_types:
            return False
        
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            return False
        
        if target_layers is None:
            return True
        
        if isinstance(target_layers, str):
            return target_layers in name
        
        if isinstance(target_layers, list):
            return any(keyword in name for keyword in target_layers)
        
        if callable(target_layers):
            return target_layers(name, layer)
        
        return True
    
    # Statistics
    total_target_layers = 0
    patched_layers = 0
    patch_report = []
    
    # Iterate over all modules
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_target_layers += 1
            
            if should_patch_layer(name, module):
                # Call the appropriate patch function based on layer type
                if isinstance(module, nn.Conv2d):
                    patched_module, success = patch_conv2d_with_lora(
                        module, rank, lora_alpha, lora_dropout, adapter_name
                    )
                    layer_type = 'Conv2d'
                else:  # nn.Linear
                    patched_module, success = patch_linear_with_lora(
                        module, rank, lora_alpha, lora_dropout, adapter_name
                    )
                    layer_type = 'Linear'
                
                if success:
                    patched_layers += 1
                    status = f"✅ PATCHED (adapter: {adapter_name})"
                else:
                    status = f"⏩ ADAPTER_EXISTS (adapter: {adapter_name})"
            else:
                status = "⏭️ SKIPPED (not in target)"
                layer_type = module.__class__.__name__
            
            # Record layer information
            layer_info = {
                'name': name,
                'type': layer_type,
                'adapter': adapter_name,
                'status': status
            }
            
            # Add layer-specific information
            if isinstance(module, nn.Conv2d):
                layer_info.update({
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels, 
                    'kernel_size': module.kernel_size
                })
            else:  # nn.Linear
                layer_info.update({
                    'in_features': module.in_features,
                    'out_features': module.out_features
                })
            
            patch_report.append(layer_info)
            
            if verbose and status != "⏩ ADAPTER_EXISTS":  # Avoid printing duplicate information
                if isinstance(module, nn.Conv2d):
                    print(f"{status}: {name} "
                          f"(in={module.in_channels}, out={module.out_channels}, "
                          f"kernel={module.kernel_size})")
                else:
                    print(f"{status}: {name} "
                          f"(in={module.in_features}, out={module.out_features})")
    
    # 打印总结
    if verbose:
        print(f"\n{'='*60}")
        print(f"LoRA Patching Summary (Adapter: {adapter_name}):")
        print(f"{'='*60}")
        print(f"Target layer types: {layer_types}")
        print(f"Total target layers found: {total_target_layers}")
        print(f"Successfully patched: {patched_layers}")
        print(f"Skipped/Already patched: {total_target_layers - patched_layers}")
        print(f"LoRA rank: {rank}")
        print(f"LoRA alpha: {lora_alpha}")
        print(f"LoRA dropout: {lora_dropout}")
        print(f"{'='*60}")
    
    return patched_layers, total_target_layers, patch_report

def get_lora_trainable_parameters(model, adapter_name=None):
    """
    Get all LoRA trainable parameters, supporting filtering by adapter
    
    Args:
        model: The model
        adapter_name: Adapter name, returns all adapter parameters if None
    """
    lora_params = []
    for name, param in model.named_parameters():
        if adapter_name is None:
            # Return all adapter parameters
            if 'lora_A' in name or 'lora_B' in name:
                lora_params.append(param)
        else:
            # Filter by adapter name
            if adapter_name == "default":
                # Default adapter uses no suffix naming
                if ('lora_A' in name and '_' not in name.split('.')[-1]) or \
                   ('lora_B' in name and '_' not in name.split('.')[-1]):
                    lora_params.append(param)
            else:
                # Specific adapters use suffix naming
                if f'lora_A_{adapter_name}' in name or f'lora_B_{adapter_name}' in name:
                    lora_params.append(param)
    
    return lora_params

def get_model_adapters(model):
    """Get all adapters information in the model"""
    adapters = {}
    for name, module in model.named_modules():
        if hasattr(module, '_lora_adapters'):
            adapters[name] = list(module._lora_adapters.keys())
    return adapters

def remove_lora_adapter(model, adapter_name):
    """
    Remove the specified LoRA adapter
    
    Args:
        model: The model
        adapter_name: Adapter name to be removed
    """
    removed_count = 0
    for name, module in model.named_modules():
        if hasattr(module, '_lora_adapters') and adapter_name in module._lora_adapters:
            config = module._lora_adapters[adapter_name]
            
            # Remove parameters
            for param_name in [config['lora_A_name'], config['lora_B_name'], config['dropout_name']]:
                if hasattr(module, param_name):
                    delattr(module, param_name)
            
            # Remove from adapter list
            del module._lora_adapters[adapter_name]
            removed_count += 1
            
            # If no adapters left, restore original forward method
            if len(module._lora_adapters) == 0:
                if hasattr(module, '_orig_forward'):
                    module.forward = module._orig_forward
                del module._lora_adapters
                del module._is_lora_patched
    
    print(f"Removed {removed_count} adapters named '{adapter_name}'")
    return removed_count

def print_model_lora_status(model, detailed=False):
    """
    Print the LoRA status of the model
    
    Args:
        model: The PyTorch model to inspect
        detailed: Whether to show detailed information for each LoRA adapter
    """
    print(f"{'Layer Name':<50} {'Type':<15} {'Adapters':<20} {'Trainable Params':<20} {'Status'}")
    print(f"{'-'*120}")
    
    total_trainable_params = 0
    total_lora_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Check if there are LoRA adapters
            has_adapters = hasattr(module, '_lora_adapters') and len(module._lora_adapters) > 0
            is_patched = hasattr(module, '_is_lora_patched')
            
            if has_adapters:
                adapter_names = list(module._lora_adapters.keys())
                adapters_str = ', '.join(adapter_names)
                status = "✅ ACTIVE"
            elif is_patched:
                adapters_str = "N/A"
                status = "⏸️ PATCHED_NO_ADAPTER"
            else:
                adapters_str = "None"
                status = "❌ NOT_PATCHED"
            
            # Calculate the number of trainable parameters
            trainable_params = 0
            if has_adapters:
                for adapter_name, config in module._lora_adapters.items():
                    lora_A = getattr(module, config['lora_A_name'])
                    lora_B = getattr(module, config['lora_B_name'])
                    trainable_params += lora_A.numel() + lora_B.numel()
                total_trainable_params += trainable_params
                total_lora_layers += 1
            
            layer_type = module.__class__.__name__
            
            print(f"{name:<50} {layer_type:<15} {adapters_str:<20} {trainable_params:<20} {status}")
            
            # Show detailed information
            if detailed and has_adapters:
                for adapter_name, config in module._lora_adapters.items():
                    lora_A = getattr(module, config['lora_A_name'])
                    lora_B = getattr(module, config['lora_B_name'])
                    print(f"    └─ Adapter: {adapter_name}")
                    print(f"       Rank: {config['rank']}, Alpha: {config['lora_alpha']}, Scaling: {config['lora_scaling']:.4f}")
                    print(f"       LoRA_A shape: {list(lora_A.shape)}, LoRA_B shape: {list(lora_B.shape)}")
    
    # Summary statistics
    print(f"{'-'*120}")
    total_model_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = (total_trainable_params / total_model_params * 100) if total_model_params > 0 else 0
    
    print(f"LoRA status summary:")
    print(f"  - Total model parameters: {total_model_params:,}")
    print(f"  - LoRA trainable parameters: {total_trainable_params:,}")
    print(f"  - LoRA parameter percentage: {trainable_percentage:.4f}%")
    print(f"  - Number of layers with LoRA: {total_lora_layers}")
    
    # Show all adapters information
    all_adapters = set()
    for name, module in model.named_modules():
        if hasattr(module, '_lora_adapters'):
            all_adapters.update(module._lora_adapters.keys())
    
    if all_adapters:
        print(f"  - Existing adapters: {', '.join(all_adapters)}")
    else:
        print(f"  - Existing adapters: None")

# Usage example
if __name__ == "__main__":
    import torchvision.models as models
    
    # Example 1: Adding multiple LoRA adapters
    print("Example 1: Adding multiple LoRA adapters to ResNet18")
    resnet = models.resnet18(pretrained=False)
    
    # First patch (using compatibility naming)
    patch_model_with_lora(resnet, rank=8, adapter_name="default", verbose=True)
    
    # Second patch (using a new adapter)
    patch_model_with_lora(resnet, rank=4, adapter_name="adapter2", verbose=True)
    
    # Third patch (attempting to add a duplicate, which will be skipped)
    patch_model_with_lora(resnet, rank=2, adapter_name="adapter2", verbose=True)
    
    # View adapters information
    adapters = get_model_adapters(resnet)
    print(f"\nAdapters in the model: {adapters}")
    
    # Get parameters for specific adapters
    default_params = get_lora_trainable_parameters(resnet, "default")
    adapter2_params = get_lora_trainable_parameters(resnet, "adapter2")
    all_params = get_lora_trainable_parameters(resnet)
    
    print(f"Number of parameters in default adapter: {sum(p.numel() for p in default_params)}")
    print(f"Number of parameters in adapter2: {sum(p.numel() for p in adapter2_params)}")
    print(f"Number of parameters in all adapters: {sum(p.numel() for p in all_params)}")
    
    # Configure optimizer (only optimize specific adapter)
    optimizer = torch.optim.Adam(adapter2_params, lr=1e-3)
    print(f"Optimizer will optimize {len(adapter2_params)} parameter groups for adapter2")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = resnet(x)
        print(f"Output shape: {output.shape}")
    
    # Example of removing an adapter
    print("\nExample of removing an adapter:")
    remove_lora_adapter(resnet, "adapter2")
    
    # View adapters information after removal
    adapters_after_removal = get_model_adapters(resnet)
    print(f"Adapters in the model after removal: {adapters_after_removal}")