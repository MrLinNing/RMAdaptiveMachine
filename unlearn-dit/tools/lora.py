import torch
import torch.nn as nn

import math
import copy

class LoRALinear(nn.Module):
    """
    This is a low-rank adapted linear layer that can be used to replace a standard linear layer.
    
    Args:
        module: The linear layer module to adapt.
        rank: The rank of the approximation.
        alpha: The alpha parameter.
    """

    def __init__(
        self,
        module: nn.Module,
        # in_dim: int,
        # out_dim: int,
        rank: int = 4,
        alpha: float = 4.0
    ):
        # ensure the module is a linear layer
        assert isinstance(module, nn.Linear), "Module must be a linear layer."

        super().__init__() # call the __init__() method of the parent class
        self.rank = rank # rank of the approximation
        self.alpha = alpha # alpha parameter
        self.scaling = self.alpha / self.rank # scaling factor
        self.in_dim = module.in_features # number of input features
        self.out_dim = module.out_features # number of output features

        # make sure that rank is at least 1
        assert self.rank >= 1, "Rank must be at least 1."

        # recreate the linear layer and freeze it
        # note: we will copy over the pretrained weights after initializing
        self.pretrained = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.pretrained.weight = nn.Parameter(module.weight.detach().clone())
        self.pretrained.bias = nn.Parameter(module.bias.detach().clone())
        self.pretrained.weight.requires_grad = False # freeze the weights
        self.pretrained.bias.requires_grad = False # freeze the bias

        # create the A and initialize with Kaiming
        self.A = nn.Linear(self.in_dim, rank, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))

        # create B and initialize with zeros
        self.B = nn.Linear(rank, self.out_dim, bias=False)
        nn.init.zeros_(self.B.weight)

        # ensure that the weights in A and B are trainable
        self.A.weight.requires_grad = True
        self.B.weight.requires_grad = True

    def forward(self, x: torch.Tensor):
        """
        Perform the forward pass of the layer.
        
        Args:
        x: The input tensor.
        """
        pretrained_out = self.pretrained(x) # get the pretrained weights
        lora_out = self.A(x) # 
        lora_out = self.B(lora_out)
        lora_out = lora_out * self.scaling
        return pretrained_out + lora_out

def freeze_parameters(model: nn.Module):
    """
    Freeze all parameters in the model.
    
    Args:
        model: The model to freeze the parameters of.
    """
    for param in model.parameters(): # iterate over the parameters of the model
        param.requires_grad = False # freeze the parameter

def unfreeze_parameters(model: nn.Module):
    """
    Unfreeze all parameters in the model.
    
    Args:
        model: The model to unfreeze the parameters of.
    """
    for param in model.parameters(): # iterate over the parameters of the model
        param.requires_grad = True # unfreeze the parameter


def get_updated_model(model: nn.Module, rank: int = 4, alpha: float = 4.0, device: str = 'cuda'):
    """
    Returns a new model with all linear layers replaced by LoRALinear layers.

    Args:
        model: The original model.
        rank: The rank of the approximation.
        alpha: The alpha parameter.
    """
    new_model = copy.deepcopy(model)
    update_model(new_model, rank, alpha, device)
    return new_model


# create a function to replace all linear layers in the the net with LoRALinear layers
def update_model(model: nn.Module, rank: int = 4, alpha: float = 4.0, device: str = 'cuda'):
    """
    Replaces all linear layers in the model with LoRALinear layers.

    Args:
    model: The model to update.
    rank: The rank of the approximation.
    alpha: The alpha parameter.
    """
    # make sure there are no LoRALinear layers in the model; return if there are
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print("Model already contains LoRALinear layers.")
            return
        
    freeze_parameters(model) # freeze all parameters in the model

    for name, module in model.named_children(): # iterate over the children of the model
        if isinstance(module, nn.Linear) and 'fc' in name: # if the module is a linear layer
            setattr(model, name, LoRALinear(module, rank, alpha)) # replace it with a LoRALinear layer
            print(f"Replaced {name} with LoRALinear layer.")
        else: # otherwise
            update_model(module, rank, alpha) # recursively call the function on the module

        # move the model to the device
        model.to(device)

    # ensure low-rank matrices are trainable
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.A.weight.requires_grad = True
            module.B.weight.requires_grad = True


# update_model_selective(model, substrings=["attn", "mlp"])
def update_model_selective(model: nn.Module, rank: int = 4, alpha: float = 4.0, device: str = 'cuda', substrings=[]):
    """
    Selectively replaces linear layers in the model with LoRALinear layers based on substrings in the layer names.

    Args:
        model: The model to update.
        rank: The rank of the approximation.
        alpha: The alpha parameter.
        device: The device to move the model to.
        substrings: A list of substrings that determine which layers should have LoRALinear applied.
    """
    # Check if there are any LoRALinear layers already
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"Model already contains a LoRALinear layer at {name}.")
            return

    # Freeze all parameters in the model
    freeze_parameters(model)

    # Define a filter function based on substrings
    def layer_filter(name):
        return any(substring in name for substring in substrings)

    # Replace selected linear layers with LoRALinear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and layer_filter(name):
            # Replace with LoRALinear layer
            setattr(model, name, LoRALinear(module, rank, alpha))
            print(f"Replaced {name} with LoRALinear layer.")

    # Move the model to the specified device
    model.to(device)

    # Ensure low-rank matrices are trainable
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.A.weight.requires_grad = True
            module.B.weight.requires_grad = True
