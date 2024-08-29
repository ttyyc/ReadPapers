import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.softmax = nn.Softmax(-1)
        self.calculate_qkv = nn.Linear(self.embedding_dim, self.embedding_dim*3)
    def forward(self, x: torch.Tensor):
        B, num_embeddings_add_cls,_ = x.shape
        qkv = self.calculate_qkv(x) 
        #tensor(batch,num_embeddings+1,3*embedding_dim)
        qkv = qkv.reshape(B, num_embeddings_add_cls, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        Q, K, V = [qkv[i] for i in range(3)]
        attention = self.softmax(torch.matmul(Q,K.transpose(-1,-2)) // (self.head_dim**0.5))
        attention = torch.matmul(attention,V)
        return attention


class MLP(nn.Module):
    def __init__(self, in_dim: int, mlp_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim    
        self.ln_1 = nn.Linear(in_dim, mlp_dim) 
        self.ln_2 = nn.Linear(mlp_dim, in_dim) 
    def forward(self, x):
        x = self.ln_1(x)
        x = self.ln_2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        embedding_dim: int,
    ):
        super(EncoderBlock, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.embedding_dim = embedding_dim
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.attention = MultiHeadAttention(self.embedding_dim,self.num_heads)
        self.mlp = MLP(self.embedding_dim, self.mlp_dim)
    def forward(self,x: torch.Tensor):
        shortcut = x
        x = self.norm(x)
        x = self.attention(x).permute(0,2,1,3).reshape(-1, 197,self.hidden_dim)
        x += shortcut
        shortcut = x
        x = self.norm(x)
        x = self.mlp(x)
        x +=  shortcut
        return x
    

class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        emdedding_dim: int,
        mlp_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = emdedding_dim
        self.mlp_dim = mlp_dim
        self.encoder_block = EncoderBlock(self.num_heads,self.hidden_dim,self.mlp_dim, self.embedding_dim)
    
    def forward(self, x: int):
        for i in range(self.num_layers):
            x = self.encoder_block(x)
        return x
    

class MlpClassificationHead(nn.Module):
    def __init__(self,embedding_dim, num_classes):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.classification = nn.Linear(self.embedding_dim,self.num_classes)

    def forward(self,x):
        x = self.classification(x)
        return x
    
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_heads: int = 12,
        num_layers: int = 12,
        hidden_dim: int = 768,
        num_classes: int = 1000,
        mlp_dim: int = 3072,
    ):
        super(VisionTransformer,self).__init__()
        assert (image_size % patch_size) == 0, f"Wrong input image size of ({image_size},{image_size})."
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.cls_token = nn.Parameter(torch.zeros(size=(1, 1, hidden_dim)))
        self.embedding_dim = (self.patch_size**2)*3
        self.mlp_dim = mlp_dim
        self.encoder = Encoder(self.num_heads,self.num_layers,self.hidden_dim,self.embedding_dim,self.mlp_dim)
        self.classification = MlpClassificationHead(self.embedding_dim, self.num_classes)
        self.image2path = nn.Conv2d(in_channels=3, out_channels=self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def tokenizer(self,x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        num_h = H // self.patch_size
        num_w = W // self.patch_size
        assert C==3, f"Wrong input channels number of {C}."
        x = self.image2path(x)
        x = x.reshape(B,self.hidden_dim,num_h*num_w).permute(0,2,1)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        return x 
    
    def forward(self,x: torch.Tensor):
        '''shape changes(take ViT-Base as a example):
            input: Nx3x224x224
            patch: Nx14x14x3x16x16
            flatten: Nx196x768
            embedding: Nx196x768
            add cls token: Nx197x768
            Encoder(Muti-head-attention): 1xNx12(heads)x197x64
            reshape: Nx197x768
            Encoder(MLP): '''
        x = self.tokenizer(x)
        x = self.encoder(x)
        x = x[:,0]
        x = self.classification(x)
        return x

def ViT_Base_16():
    '''build vision transformer base model'''
    return VisionTransformer(224,16,12,12,768,1000,302)


if __name__=="__main__":
    model = ViT_Base_16()
    x = torch.ones(size=(640,3,224,224))
    out = model(x)
    print(out.shape)   






