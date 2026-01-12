import torch
import torch.nn as nn

class ImageToPatches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.P = patch_size

    def forward(self, x):
        P = self.P
        B,C,H,W = x.shape                       # [B,C,H,W]                 4D Image
        x = x.reshape(B,C, H//P, P , W//P, P)   # [B,C, H//P, P, W//P, P]   6D Patches
        x = x.permute(0,2,4, 1,3,5)             # [B, H//P, W//P, C, P, P]  6D Swap Axes
        x = x.reshape(B, H//P * W//P, C*P*P)    # [B, H//P * W//P, C*P*P]   3D Patches
                                                # [B, n_tokens, n_pixels]
        return x

class PerPatchMLP(nn.Module):
    def __init__(self, n_pixels, n_channel):
        super().__init__()
        self.mlp = nn.Linear(n_pixels, n_channel, bias=False)  # n_pixels = n_image_channel * patch_size**2

    def forward(self, x):      
        return self.mlp(x)  # x*w:  [B, n_tokens, n_pixels] x [n_pixels, n_channel]   
                            #       [B, n_tokens, n_channel]   
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))  

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True)) + self.eps
        return self.gamma * (x / rms)

class TokenMixingMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_hidden):
        super().__init__()
        # self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.layer_norm = RMSNorm(n_channel)
        self.mlp1 = nn.Linear(n_tokens, n_hidden, bias=False)  # n_tokens = H//P * W//P
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(n_hidden, n_tokens, bias=False)  # n_tokens = H//P * W//P

    def forward(self, X):
        z = self.layer_norm(X)                  # z:    [B, n_tokens, n_channel]
        z = z.permute(0, 2,1)                   # z:    [B, n_channel, n_tokens]
        z = self.gelu(self.mlp1(z))             # z:    [B, n_channel, n_hidden] 
        z = self.mlp2(z)                        # z:    [B, n_channel, n_tokens]
        z = z.permute(0, 2,1)                   # z:    [B, n_tokens, n_channel]
        U = X + z                               # U:    [B, n_tokens, n_channel]
        return U



class ChannelMixingMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_hidden):
        super().__init__()
        # self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.layer_norm = RMSNorm(n_channel)
        self.mlp3 = nn.Linear(n_channel, n_hidden, bias=False)
        self.gelu = nn.GELU()
        self.mlp4 = nn.Linear(n_hidden, n_channel, bias=False)

    def forward(self, U):
        z = self.layer_norm(U)                  # z: [B, n_tokens, n_channel]
        z = self.gelu(self.mlp3(z))             # z: [B, n_tokens, n_hidden]
        z = self.mlp4(z)                        # z: [B, n_tokens, n_channel]
        Y = U + z                               # Y: [B, n_tokens, n_channel]
        return Y
    


class OutputMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_output):
        super().__init__()
        # self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.layer_norm = RMSNorm(n_channel)
        self.out_mlp = nn.Linear(n_channel, n_output, bias=False)  # n_output = number of classes

    def forward(self, x):
        x = self.layer_norm(x)                  # x: [B, n_tokens, n_channel]
        x = x.mean(dim=1)                       # x: [B, n_channel] 
        return self.out_mlp(x)                  # x: [B, n_output]

class MLP_Mixer(nn.Module):
    def __init__(self, n_layers, n_channel, n_hidden, n_output, image_size, patch_size, n_image_channel):
        super().__init__()

        n_tokens = (image_size // patch_size)**2
        n_pixels = n_image_channel * patch_size**2

        self.ImageToPatch = ImageToPatches(patch_size = patch_size)
        self.PerPatchMLP = PerPatchMLP(n_pixels, n_channel)
        self.MixerStack = nn.Sequential(*[
            nn.Sequential(
                TokenMixingMLP(n_tokens, n_channel, n_hidden),
                ChannelMixingMLP(n_tokens, n_channel, n_hidden)
            ) for _ in range(n_layers)
        ])
        self.OutputMLP = OutputMLP(n_tokens, n_channel, n_output)

    def forward(self, x):
        x = self.ImageToPatch(x)
        x = self.PerPatchMLP(x)
        x = self.MixerStack(x)
        return self.OutputMLP(x)
    

class MLP_Mixer_plot(nn.Module):
    def __init__(self, n_layers, n_channel, n_hidden, n_output, image_size, patch_size, n_image_channel):
        super().__init__()
        
        self.n_channel = n_channel
        self.patch_size = patch_size
        self.image_size = image_size
        
        n_tokens = (image_size // patch_size)**2
        n_pixels = n_image_channel * patch_size**2

        self.ImageToPatch = ImageToPatches(patch_size=patch_size)
        self.PerPatchMLP = PerPatchMLP(n_pixels, n_channel)
        self.MixerStack = nn.Sequential(*[
            nn.Sequential(
                TokenMixingMLP(n_tokens, n_channel, n_hidden),
                ChannelMixingMLP(n_tokens, n_channel, n_hidden)
            ) for _ in range(n_layers)
        ])
        self.OutputMLP = OutputMLP(n_tokens, n_channel, n_output)

    def forward(self, x):
        x = self.ImageToPatch(x)
        x = self.PerPatchMLP(x)
        x = self.MixerStack(x)
        return self.OutputMLP(x)