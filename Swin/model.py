import torch
import torch.nn as nn

class PatchPartition(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size(int): Image size. Default: 224.
        patch_size(int): Patch token size. Default: 4.
        in_chans(int): Number of input image channels. Default: 3.
        embed_dim(int): Number of linear projection output channels. Default: 96.
        norm_layer(nn.Module): Normalization layer. Default: None.
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 embed_dim: int = 96,
                 norm_layer: nn.Module = None
                 ):
        super().__init__()
        self.patches_resolution = img_size // patch_size
        self.num_patches = self.patches_resolution * self.patches_resolution
        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape 
        x = self.projection(x).flatten(2).transpose(-2,-1)
        if self.norm is not None:
            x = self.norm(x)
        return x
    

def window_partiton(x, window_size: int):
    r"""Window partiton
    
    Args:
        x(tensor): (B, C, H, W)
        window_size(int): window size.

    Returns:
        windows(tensoor): (B*num_windows, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, H // window_size, W // window_size, window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    r"""Window reserve
    Args:
        windows(tensor): (B*num_windows, window_size, window_size, C)
        window_size(int): Window size.
        H(int): Height of image.
        W(int): Width of image.

    Returns:
        x(tensor): (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttenton(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim(int): Number of input channels.
        window_size(int): Window size.
        num_heads(int): Number of attention heads.
        qkv_bias(bool): If True, add a learnable bias to query, keey, value
        qk_scale(float): If set, override default qk scale of head_dim ** -0.5
        attn_drop(float): Dropout ratio of attention  weights. Default: 0.0.
        proj_drop(float): Dropout ratio of output. Default: 0.0.
    """
    def __init__(self,
                 dim: int,
                 window_size: int,
                 num_heads: int,
                 qkv_bias: bool,
                 qk_scale: float,
                 attn_drop: float,
                 proj_drop: float,
                 ):
        super().__init__()
        


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block
    
    Args:

    """
    

class SwinTransformer(nn.Module):
    r""" Swin Transformer for classification
    
    Args:
        img_size(int): Image size. Default: 224.
        patch_size(int): Patch tolen size. Default: 4.
        in_chans(int): Number of input image size. Default: 3.
        embed_dim(int): Number of linear projection output channels. Default: 96.
        num_classes(int): Number of classes for classification head. Default: 1000.
        depths(tuple(int)): Depth of each Swin Transformer layer.
        num_heads(tuple(int)): Number of attention heads in different layers.
        window_size(int): Window size. Default: 7.
        ape(bool): Add absolute position embedding to the patch embedding. Default: False
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 embed_dim: int = 96,
                 num_classes: int = 1000,
                 depth: tuple[int] = [2,2,6,2],
                 num_heas: tuple[int] = [3,6,12,24],
                 window_size: int = 7,
                 norm_layer: nn.Module = None,
                 ape: bool = False,
                 ):
        super().__init__()
        self.patch_partition = PatchPartition(img_size, patch_size, in_chans, embed_dim, norm_layer)
        num_patches = self.patch_partition.num_patches
        self.ape = ape
        if self.ape:
            self.absolute_position_embedding = nn.Parameter(torch.zeros(size=(1, num_patches, embed_dim)))





if __name__=="__main__":
    # Patch Partition Test block
    x1 = torch.randn(size=(2,3,224,224))
    patch_partition = PatchPartition(norm_layer=nn.LayerNorm)
    x1 = patch_partition(x1)
    print("Patch Partition Test: ",x1.shape)

    # Window Partition and Reverse Test block
    x2 = torch.randn(size=(2, 96, 56, 56))
    _, _, height, width = x2.shape
    x2 = window_partiton(x2, window_size=7)
    print("Window Partition Test: ", x2.shape)
    x2 = window_reverse(x2, window_size=7, H=height, W=width)
    print("Window Reverse Test: ",x2.shape)

    # Window attention (including W-MSA and SW-MSA) Test Block
     


        

