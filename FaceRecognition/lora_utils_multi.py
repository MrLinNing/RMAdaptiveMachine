import torch
import torch.nn as nn
import copy
import math

# Assuming model_mixer contains the original MLP_Mixer, TokenMixingMLP, ChannelMixingMLP, OutputMLP
from model_mixer import TokenMixingMLP, ChannelMixingMLP, OutputMLP

def freeze_parameters(model: nn.Module):
    """
    Freeze all parameters in the model.
    """
    for param in model.parameters():
        param.requires_grad = False

def print_params(model: nn.Module):
    """
    Print the number of trainable and untrainable parameters in the model.
    """
    trainable = 0
    untrainable = 0
    trainable_size = 0
    lora_trainable = {}

    print("Layers:")
    for name, param in model.named_parameters():
        is_lora_param = 'lora_branches' in name
        lora_branch_name = None
        if is_lora_param:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'lora_branches':
                    lora_branch_name = parts[i+1]
                    break
            if lora_branch_name and lora_branch_name not in lora_trainable:
                lora_trainable[lora_branch_name] = 0

        print(f"- {name} of size {param.size()} -> {'trainable' if param.requires_grad else 'untrainable'}")
        if param.requires_grad:
            trainable += param.numel()
            trainable_size += param.numel() * param.element_size()
            if lora_branch_name:
                lora_trainable[lora_branch_name] += param.numel()
        else:
            untrainable += param.numel()

    print(f"\nTrainable parameters: {trainable}")
    print(f"Untrainable parameters: {untrainable}")
    print(f"Total parameters: {trainable + untrainable}")
    print(f"Percent trainable: {100 * trainable / (trainable + untrainable)}%")
    print(f"Size of trainable parameters: {trainable_size / 1e6:.2f} mb")
    if lora_trainable:
        print("\nTrainable parameters by LoRA branch:")
        for branch_name, count in lora_trainable.items():
            print(f"  {branch_name}: {count}")

class LoRALinear_Attention(nn.Module):
    def __init__(
        self,
        module: nn.Linear,
        rank: int = 16,
        alpha: float = 4.0,
        mode: str = "attention"
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_dim = module.in_features
        self.out_dim = module.out_features
        self.mode = mode
        self.lora_branches = nn.ModuleDict()  # Store multiple LoRA branches
        self.active_lora = None  # Current active LoRA branch

        if self.mode not in ["attention", "add"]:
            raise ValueError("Mode must be 'attention' or 'add'")

        # Frozen pretrained linear layer
        self.pretrained = nn.Linear(self.in_dim, self.out_dim, bias=module.bias is not None)
        self.pretrained.weight = nn.Parameter(module.weight.detach().clone())
        self.pretrained.weight.requires_grad = False
        if module.bias is not None:
            self.pretrained.bias = nn.Parameter(module.bias.detach().clone())
            self.pretrained.bias.requires_grad = False

        self.is_lora = True

    def add_lora_branch(self, name: str, rank: int = None):
        """Add a new LoRA branch with the given name."""
        if name in self.lora_branches:
            raise ValueError(f"LoRA branch '{name}' already exists.")
        rank = rank or self.rank  # Use provided rank or default
        A = nn.Linear(self.in_dim, rank, bias=False)
        B = nn.Linear(rank, self.out_dim, bias=False)
        nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
        nn.init.zeros_(B.weight)
        self.lora_branches[name] = nn.ModuleDict({'A': A, 'B': B})
        print(f"Added LoRA branch: {name} with rank={rank}")

    def set_active_lora(self, lora_name: str):
        """Set the active LoRA branch for the forward pass."""
        if lora_name is not None and lora_name not in self.lora_branches:
            raise ValueError(f"LoRA branch '{lora_name}' not found. Available: {list(self.lora_branches.keys())}")
        self.active_lora = lora_name

    def set_trainable_lora(self, lora_names: list):
        """Set specified LoRA branches as trainable."""
        invalid = [name for name in lora_names if name not in self.lora_branches]
        if invalid:
            raise ValueError(f"Invalid LoRA branches: {invalid}. Available: {list(self.lora_branches.keys())}")
        for param in self.parameters():
            param.requires_grad = False
        for name in lora_names:
            branch = self.lora_branches[name]
            branch['A'].weight.requires_grad = True
            branch['B'].weight.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pretrained_out = self.pretrained(x)
        if self.active_lora is None:
            return pretrained_out
        branch = self.lora_branches[self.active_lora]
        lora_out = branch['B'](branch['A'](x)) * self.alpha
        if self.mode == "attention":
            return pretrained_out * torch.sigmoid(lora_out)
        else:  # mode == "add"
            return pretrained_out + lora_out

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, mode={self.mode}, active_lora={self.active_lora}, branches={list(self.lora_branches.keys())}"

def add_lora_to_model(
    model: nn.Module,
    target_modules: tuple = (TokenMixingMLP, ChannelMixingMLP, OutputMLP),
    rank: int = 16,
    alpha: float = 4.0,
    mode: str = "add",
    lora_name: str = "lora",
    device: str = 'cuda'
):
    """
    Adds a new LoRA branch to the model in-place.
    """
    freeze_parameters(model)

    def process_module(parent, module, name_prefix: str = ""):
        if name_prefix in ['unknown_class_head1', 'unknown_class_head2']:
            return
        if isinstance(module, target_modules):
            print(f"Processing target module: {name_prefix}")
            for child_name, child_module in module.named_children():
                full_name = f"{name_prefix}.{child_name}" if name_prefix else child_name
                if isinstance(child_module, nn.Linear) and not isinstance(child_module, LoRALinear_Attention):
                    new_lora_layer = LoRALinear_Attention(child_module, rank, alpha, mode=mode)
                    new_lora_layer.add_lora_branch(lora_name, rank=rank)
                    setattr(module, child_name, new_lora_layer)
                    print(f"  Created LoRALinear_Attention with branch '{lora_name}' at: {full_name}")
                elif isinstance(child_module, LoRALinear_Attention):
                    child_module.add_lora_branch(lora_name, rank=rank)
                    print(f"  Added branch '{lora_name}' to existing LoRALinear_Attention at: {full_name}")
                elif isinstance(child_module, nn.Module):
                    process_module(module, child_module, full_name)
        elif isinstance(module, nn.Module):
            for child_name, child_module in module.named_children():
                full_name = f"{name_prefix}.{child_name}" if name_prefix else child_name
                process_module(module, child_module, full_name)

    for name, child in model.named_children():
        process_module(model, child, name)
    model.to(device)

def get_updated_model(
    model: nn.Module,
    target_modules: tuple = (TokenMixingMLP, ChannelMixingMLP, OutputMLP),
    rank: int = 16,
    alpha: float = 4.0,
    mode: str = "add",
    lora_name: str = "lora",
    device: str = 'cuda'
):
    """
    Returns a new model with a LoRA branch added.
    """
    new_model = copy.deepcopy(model)
    add_lora_to_model(new_model, target_modules, rank, alpha, mode, lora_name, device)
    return new_model