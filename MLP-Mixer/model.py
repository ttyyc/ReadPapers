import torch
import torch.nn as nn

class Tokenizer(nn.Module):
    '''split original image to many patches'''
    def __init__(
        self,
        patch_size: int,
        num_patches: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
    
    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape(N, C, self.num_patches, self.patch_size, self.patch_size)
        x = x.reshape(N,self.num_patches,-1)
        return x


class MLP(nn.Module):
    '''MLP block for MLP-Mixer architecture'''
    def __init__(
        self,
        in_features: int,
        mlp_dim: int,
        out_features: int
    ):
        super().__init__()
        self.in_features = in_features
        self.mlp_dim = mlp_dim
        self.out_features = out_features
        self.fc_1 = nn.Linear(self.in_features, self.mlp_dim)
        self.gelu = nn.GELU()
        self.fc_2 = nn.Linear(self.mlp_dim, self.out_features)
    
    def forward(self,x):
        x = self.fc_1(x)
        x = self.gelu(x)
        x = self.fc_2(x)
        return x
    

class MixerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        sequence_length: int = 196,
        tokens_mlp_dim: int = 384,
        channels_mlp_dim: int = 3072,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.tokens_mixer = MLP(in_features=sequence_length, mlp_dim=tokens_mlp_dim, out_features=sequence_length)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.channels_mixer = MLP(in_features=hidden_dim, mlp_dim=channels_mlp_dim, out_features=hidden_dim)

    def forward(self,x):
        x_identity = x
        x = self.ln_1(x).transpose(-1,-2)
        x = self.tokens_mixer(x).transpose(-1,-2)
        x += x_identity

        x_identity = x 
        x = self.ln_2(x)
        x = self.channels_mixer(x)
        x += x_identity
        del x_identity
        return x


class Classifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self,x):
        x = self.fc(x)
        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        hidden_dim: int = 768,
        sequence_length: int = 196,
        tokens_mlp_dim: int = 384,
        channels_mlp_dim: int = 3072,
        num_layers: int = 12,
        num_class: int = 1000,
    ):
        super().__init__()
        self.tokenizer = Tokenizer(patch_size, sequence_length)
        self.mlp_block = MixerBlock(patch_size,hidden_dim, sequence_length, tokens_mlp_dim, \
                                    channels_mlp_dim, num_layers, num_class)
        
        self.classifier = Classifier(hidden_dim, num_class)

    def forward(self,x):
        x = self.tokenizer(x)
        x = self.mlp_block(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x
    
def mixer_s32(
    patch_size: int = 32,
    hidden_dim: int = 512,
    sequence_length: int = 49,
    tokens_mlp_dim: int = 256,
    channels_mlp_dim: int = 2048,
    num_layers: int = 8,
    num_class: int = 1000
    ):
    return MLPMixer(patch_size, hidden_dim, sequence_length, tokens_mlp_dim, channels_mlp_dim, num_layers, num_class)


if __name__=='__main__':
    model = mixer_s32()
    x = torch.ones(size=(1,3,224,224))
    out = model(x)
    print(out.shape)