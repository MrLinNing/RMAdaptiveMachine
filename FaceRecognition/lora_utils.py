import torch
import matplotlib as mpl
import torch.nn as nn
import copy
import math
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from model_mixer import TokenMixingMLP, ChannelMixingMLP, OutputMLP


def freeze_parameters(model: nn.Module):
    """
    Freeze all parameters in the model.
    
    Args:
        model: The model to freeze the parameters of.
    """
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_parameters(model: nn.Module):
    """
    Unfreeze all parameters in the model.
    
    Args:
        model: The model to unfreeze the parameters of.
    """
    for param in model.parameters():
        param.requires_grad = True

def print_params(model: nn.Module):
    """
    Print the number of trainable and untrainable parameters in the model.
    
    Args:
        model: The model to print the parameters of.
    """
    trainable = 0
    untrainable = 0
    trainable_size = 0

    print("Layers:")
    for name, param in model.named_parameters():
        print(f"- {name} of size {param.size()} -> {'trainable' if param.requires_grad else 'untrainable'}")
        if param.requires_grad:
            trainable += param.numel()
            trainable_size += param.numel() * param.element_size()
        else:
            untrainable += param.numel()

    print(f"\nTrainable parameters: {trainable}")
    print(f"Untrainable parameters: {untrainable}")
    print(f"Total parameters: {trainable + untrainable}")
    print(f"Percent trainable: {100 * trainable / (trainable + untrainable)}%")
    print(f"Size of trainable parameters: {trainable_size / 1e6:.2f} mb")


class LoRALinear_Attention(nn.Module):
    def __init__(
        self,
        module: nn.Linear,
        rank: int = 16,
        alpha: float = 4.0,
        mode: str = "attention"  # New parameter to specify mode
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_dim = module.in_features
        self.out_dim = module.out_features
        self.mode = mode  # Store the mode ("attention" or "add")
        
        # Validate mode
        if self.mode not in ["attention", "add"]:
            raise ValueError("Mode must be 'attention' or 'add'")
        
        # Create frozen pretrained layer (including bias)
        self.pretrained = nn.Linear(self.in_dim, self.out_dim, bias=module.bias is not None)
        self.pretrained.weight = nn.Parameter(module.weight.detach().clone())
        self.pretrained.weight.requires_grad = False
        if module.bias is not None:
            self.pretrained.bias = nn.Parameter(module.bias.detach().clone())
            self.pretrained.bias.requires_grad = False
        
        # Encoder and decoder for LoRA
        self.A = nn.Linear(self.in_dim, rank, bias=False)
        self.B = nn.Linear(rank, self.out_dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        
        self.is_lora = True  # Identification flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pretrained_out = self.pretrained(x)
        lora_out = self.B(self.A(x)) * self.alpha
        if self.mode == "attention":
            return pretrained_out * torch.sigmoid(lora_out)
        else:  # mode == "add"
            return pretrained_out + lora_out

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, mode={self.mode}"

def get_updated_model(
    model: nn.Module,
    target_modules: tuple = (TokenMixingMLP, ChannelMixingMLP),
    rank: int = 16,
    alpha: float = 4.0,
    mode: str = "attention",
    device: str = 'cuda'
):
    """
    Returns a new model with LoRA added to specified target modules.
    
    Args:
        model: Original model
        target_modules: Tuple of module types to apply LoRA to
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        mode: LoRA mode ("attention" or "add")
        device: Device to move model to
    """
    new_model = copy.deepcopy(model)
    update_model(new_model, target_modules, rank, alpha, mode, device)
    return new_model

def update_model(
    model: nn.Module,
    target_modules: tuple = (TokenMixingMLP, ChannelMixingMLP),
    rank: int = 4,
    alpha: float = 4.0,
    mode: str = "attention",
    device: str = 'cuda'
):
    """
    Adds LoRA to linear layers within specified target modules.
    
    Args:
        model: Original model
        target_modules: Tuple of module types to apply LoRA to
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        mode: LoRA mode ("attention" or "add")
        device: Device to move model to
    """
    # Check if model already has LoRA layers
    for module in model.modules():
        if isinstance(module, LoRALinear_Attention):
            print("Model already contains LoRALinear_Attention layers.")
            return

    freeze_parameters(model)  # Freeze all original parameters

    def process_module(parent, module, name_prefix: str = ""):
        # Skip specific heads if present
        if name_prefix in ['unknown_class_head1', 'unknown_class_head2']:
            return

        # If we find a target module type
        if isinstance(module, target_modules):
            print(f"Processing target module: {name_prefix}")
            for child_name, child_module in module.named_children():
                full_name = f"{name_prefix}.{child_name}" if name_prefix else child_name
                if isinstance(child_module, nn.Linear):
                    new_lora_layer = LoRALinear_Attention(child_module, rank, alpha, mode=mode)
                    setattr(module, child_name, new_lora_layer)
                    print(f"  Replaced linear layer: {full_name} with mode={mode}")
                elif isinstance(child_module, nn.Module):
                    process_module(module, child_module, full_name)
        
        # Continue recursion for non-target modules
        elif isinstance(module, nn.Module):
            for child_name, child_module in module.named_children():
                full_name = f"{name_prefix}.{child_name}" if name_prefix else child_name
                process_module(module, child_module, full_name)

    # Start processing from the top-level model
    for name, child in model.named_children():
        process_module(model, child, name)

    model.to(device)

