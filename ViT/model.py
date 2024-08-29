import torch
import torch.nn as nn
import torchvision
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
        attention = torch.matmul((self.softmax(torch.matmul(Q,K.T) // (self.head_dim**0.5))),V)
        return attention


class MLP(nn.Module):
    def __init__(self, in_dim: int, mlp_dim: int):
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim    
        self.mlp = nn.Linear(in_dim, mlp_dim)    
    def forward(self, x):
        x = self.mlp(x)
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
        self.norm = nn.LayerNorm()
        self.attention = MultiHeadAttention(self.embedding_dim,self.num_heads)
        self.mlp = MLP(self.embedding_dim, self.hidd)
    def forward(self,x: torch.Tensor):
        shortcut = x
        x = self.norm(x)
        x = self.attention(x) + shortcut
        shortcut = x
        x = self.norm(x)
        x = self.mlp(x) + shortcut
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
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = emdedding_dim
        self.mlp_dim = mlp_dim
        self.encoder_block = EncoderBlock(self.num_heads,self.hidden_dim,self.mlp_dim)
    
    def forward(self, x: int):
        for i in range(self.num_layers):
            x = self.encoder_block(x)
        return x
    

class MlpClassificationHead(nn.Module):
    def __init__(self,embedding_dim, num_classes):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.classification = (self.embedding_dim,self.num_classes)

    def forward(self,x):
        x = self.classification(x)
        return x
    
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        num_classes: int,
        mlp_dim: int,
    ):
        super(VisionTransformer,self).__init__()
        assert (image_size % patch_size) == 0, f"Wrong input image size of ({image_size},{image_size})."
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.cls_token = nn.parameter(torch.zeros(size=(1, 1, hidden_dim)))
        self.embedding_dim = (self.patch_size**2)*3
        self.mlp_dim = mlp_dim
        self.encoder = Encoder(self.num_heads,self.num_layers,self.hidden_dim,self.embedding_dim,self.mlp_dim)
        self.classification = MlpClassificationHead(self.embedding_dim, self.num_classes)

    def tokenizer(self,x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        num_h = H // self.patch_size
        num_w = W // self.patch_size
        assert C==3, f"Wrong input channels number of {C}."
        x = nn.Conv2d(in_channels=C, out_channels=self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        x = x.reshape(B,self.hidden_dim,num_h*num_w).permute(0,2,1)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        return x 
    
    def forward(self,x: torch.Tensor):
        x = self.tokenizer(x)
        x = self.encoder(x)
        x = self.classification(x)
        return x

if __name__=="__main__":
    model = VisionTransformer(224, 16, 196, 6, 12, )    






