import torch
import torch.nn as nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t,t,t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super.__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# class FeedForward(nn.Module):


class DenseForward(nn.Module):
    def __init__(self, dim, hidden_dim, outdim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, outdim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Dense_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=None):
        super().__init__()
        inner_dim = dim_head*heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1) #[2, 512, 32] -> [2, 512, 96] -> [2, 512, 32]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads ), qkv) #[2,8,512,4]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale #[b,h,n,n]
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class DensePreConv_AttentionBlock(nn.Module):
    def __init__(self, out_channels, height, width, growth_rate=32, depth=4, heads=8, dropout=0.5, attention=Dense_Attention):
        super().__init__()
        mlp_dim = growth_rate * 2
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Linear(out_channels+i*growth_rate, growth_rate),
                # PreNorm(growth_rate,
                #         attention(growth_rate, heads=heads, dim_head=(growth_rate) // heads, dropout=dropout,
                #                   num_patches=(height, width))),
                PreNorm(growth_rate, attention(growth_rate, heads=heads, dim_head=(growth_rate) // heads, dropout=dropout,
                                               num_patches=(height, width))),
                PreNorm(growth_rate, DenseForward(growth_rate, mlp_dim, growth_rate, dropout=dropout))
            ]))
        self.out_layer = DenseForward(out_channels+depth*growth_rate, mlp_dim, out_channels, dropout=dropout)
    def forward(self, x):
        features = [x] #[2, 512, 64]
        # print(len(self.layers)) #4(depth)
        for l, attn, ff in self.layers:
            x = torch.cat(features, 2)
            x = l(x)
            x = attn(x) + x
            x = ff(x) + x
            features.append(ff(x))
        x = torch.cat(features, 2)
        x = self.out_layer(x)
        return x

class Dense_TransformerBlock():
    def __int__(self, in_channels, out_channels, image_size, growth_rate=32,
                patch_size=16, depth=6, heads=8, dropout=0.5, attention = DensePreConv_AttentionBlock):
        super().__init__()
        image_depth, image_height, image_width = pair(image_size)
        patch_depth, patch_height, patch_width = pair(patch_size)
        self.outsize = (image_depth // patch_size, image_height // patch_size, image_width // patch_size) #(4,8,16)
        d = image_depth // patch_depth
        h = image_height // patch_height
        w = image_width // patch_width
        num_patches = d * h * w
        mlp_dim = out_channels * 2
        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, out_channels))
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            self.blocks.append(nn.ModuleList([
                attention(out_channels, height=h, width=w, growth_rate=growth_rate)
                               ]))
        # self.blocks = nn.ModuleList([attention(out_channels, height=h, width=w, growth_rate=growth_rate) for i in range(depth)])
        self.re_patch_embedding = nn.Sequential(
            Rearrange('b (d h w) (c) -> b c (d) (h) (w)', h = h, w = w)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.patch_embeddings(img) # [2, 1, 64, 128, 256] -> [2, 128, 4, 8, 16]
        x = x.flatten(2) # [2, 128, 4, 8, 16] -> [2, 128, 4*8*16]
        x = x.transpose(-1,-2) # [2, 4*8*16, 128]
        embeddings = x + self.position_embeddings
        x = self.dropout(embeddings)

        for block in self.blocks:
            x = block(x)
        # x: [2, 512, 64]

        x = self.re_patch_embedding(x) #[2, 128, 4, 8, 16]
        return F.interpolate(x, self.outsize) #[4, 8, 16]


# if __name__ == '__main__':
    # transformer_depth = 12
    # # in_channels=2, n_cls=4, image_size=(64, 128, 256), n_filters=16
    # Trans_encoder = Dense_TransformerBlock(in_channels=1,out_channels=4 * n_filters,image_size=image_size,
    #             patch_size=16,depth=transformer_depth//4,attention=DensePreConv_AttentionBlock)
    # output = torch.rand((2,128,4,8,16))