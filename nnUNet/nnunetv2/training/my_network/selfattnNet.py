import torch
import einops
from torch import nn
from typing import Tuple, Union, List, Type
from timm.models.layers import DropPath, trunc_normal_
from nnunetv2.training.my_network.UNetRPP.dynunet_block import UnetOutBlock, UnetResBlock
from nnunetv2.training.my_network.UNetRPP.model_components import UnetrPPEncoder, UnetrUpBlock
from nnunetv2.training.my_network.UNetRPP.dynunet_block import get_conv_layer, UnetResBlock
from nnunetv2.training.my_network.UNetRPP.transformer_block import TransformerBlock
from nnunetv2.training.my_network.UNetRPP.layers import LayerNorm
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
# from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from nnunetv2.training.my_network.UNetDecoder import UNetDecoder, UNetDecoder2, UNetDecoder3, UNetDecoder4, UNetDecoder5, UNetDecoder6, Cross_Attention, Self_Attention
from monai.networks.layers.utils import get_norm_layer
from monai.networks.nets import SwinUNETR

# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=8, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super.__init__()
#         self.num_heads = num_heads
#         head_dim = dim//num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.wq = nn.Linear(dim, dim, bias=qkv_bias)
#         self.wk = nn.Linear(dim, dim, bias=qkv_bias)
#         self.wv = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B,N,C = x.shape
#         q = self.wq()

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=9, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim// num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale # self-attention
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1,2).reshape(B, N, C) #attn weights 得到 features
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ration, qkv_bias):
#         super().__init__()
#         self.attn = CrossAttention()
#         self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
#         self.has_mlp = has_mlp
#         if has_mlp:
#             self.norm2 = norm_layer(dim)
#             mlp_hidden_dim = int(dim * mlp_ratio)
#             self.mlp = MLP
#
#     def forward(self, x):
#         if self.has_mlp:
#             x =
#             x = x + self.drop_path(self.mlp(self.norm2(x)))

# class PatchEmbed(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_chans, embed_dim, multi_conv=False):
#         super().__init__()
#         img_size =
#         patch_size =
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
#         self.proj = nn.Conv3d(in_channels=in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#
#     def forward(self, x):
#         B,C,H,W,D = x.shape
#         assert H == self.img_size and W == self.img_size[1]
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x

# modified based on v5; no decoder fusion; larger feature size reserved
class HybridNet_v6(nn.Module):
    def __init__(self,
                 conv_bias,
                 norm_op,
                 norm_op_kwargs,
                 dropout_op,
                 dropout_op_kwargs,
                 nonlin,
                 nonlin_kwargs,
                 features_per_stage,
                 kernel_sizes,
                 strides,
                 n_conv_per_stage,
                 n_conv_per_stage_decoder,
                 input_channels: int = 2,
                 out_channels: int = 4,
                 n_stages: int = 5,
                 conv_op=nn.Conv3d,
                 pool: str = 'conv',
                 feature_size: int = 16,
                 hidden_size: int = 256,
                 num_heads: int = 4,
                 norm_name: Union[Tuple, str] = "instance",
                 dropout_rate: float = 0.0,
                 depths=None,
                 dims=None,
                 do_ds=True,
                 ):
        super().__init__()
        # self.input_channels = input_channels
        self.input_channels = 1
        self.num_classes = out_channels
        if depths is None:
            self.depths = [3, 3, 3, 3]
        self.num_heads = num_heads
        self.do_ds = do_ds
        self.spe_encoder1 = PlainConvEncoder(self.input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                                             strides,
                                             n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                             dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                             nonlin_first=False)
        self.spe_encoder2 = PlainConvEncoder(self.input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                                             strides,
                                             n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                             dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                             nonlin_first=False)
        self.share_encoder = UnetrPPEncoderv1(input_size=[32 * 64 * 128, 16 * 32 * 64, 8 * 16 * 32, 4 * 4 * 8],
                                              dims=[32, 64, 128, 512], depths=self.depths, num_heads=self.num_heads,
                                              in_channels=2,
                                              hidden_size=512)
        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=8 * 16 * 32,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 32 * 64,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 64 * 128,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=64 * 128 * 256,
            conv_decoder=True,
        )
        self.fusion = EnhancedFeature(in_chans=256)
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)
        self.projection1 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=3, padding=1, bias=True),
                                         nn.Conv3d(256, 256, kernel_size=1))
        self.projection2 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=3, padding=1, bias=True),
                                         nn.Conv3d(256, 256, kernel_size=1))
        self.projection3 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=3, padding=1, bias=True),
                                         nn.Conv3d(256, 256, kernel_size=1))
        # self.projection = nn.Conv3d(512, 256, kernel_size=3, padding=1, bias=True # a simple dim reduction 512->256
        # self.projection = nn.Conv3d(512, 256, kernel_size=1))
        self.f1 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, padding=1, bias=True),
                                nn.Conv3d(64, 16, kernel_size=1))
        self.f2 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, padding=1, bias=True),
                                nn.Conv3d(64, 16, kernel_size=1))
        self.g1 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                # nn.BatchNorm1d(1024),
                                nn.ReLU(),
                                nn.Linear(in_features=1024, out_features=256))
        self.g2 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                # nn.BatchNorm1d(1024),
                                nn.ReLU(),
                                nn.Linear(in_features=1024, out_features=256))
        self.t1 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, padding=1, bias=True),
                                nn.Conv3d(64, 16, kernel_size=1))
        self.t2 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, padding=1, bias=True),
                                nn.Conv3d(64, 16, kernel_size=1))
        self.classifier1 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(in_features=1024, out_features=256),
                                         nn.Linear(in_features=256, out_features=100),
                                         nn.BatchNorm1d(100),
                                         nn.ReLU(),
                                         nn.Linear(in_features=100, out_features=1),
                                         # nn.Sigmoid()
                                         )
        self.classifier2 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(in_features=1024, out_features=256),
                                         nn.Linear(in_features=256, out_features=100),
                                         nn.BatchNorm1d(100),
                                         nn.ReLU(),
                                         nn.Linear(in_features=100, out_features=1),
                                         # nn.Sigmoid()
                                         )  # use cross entropy lossr(data) # get shared features for T2 and TOF simultaneously [B,256,4,8,8]
        self.fusion_layer1 = Fusion_layer(128)
        self.fusion_layer2 = Fusion_layer(64)
        self.fusion_layer3 = Fusion_layer(32)

    def forward(self, data):
        fea_1_spe_all = self.spe_encoder1(data[:, 0:1, ...])  # get specific features for T2 [B,256,4,4,8]
        fea_1_spe = fea_1_spe_all[-1]
        fea_2_spe_all = self.spe_encoder2(data[:, 1:2, ...])  # get specific features for TOF
        fea_2_spe = fea_1_spe_all[-1]
        # fea_sha,  = self.share_encoder(data) # get shared features for T2 and TOF simultaneously
        # fea_1_sha, fea_2_sha
        # fea_1_sha, hidden_states1 = self.share_encoder(data[:, 0:1, ...])
        # fea_2_sha, hidden_states2 = self.share_encoder(data[:, 1:2, ...])
        fea_sha, hidden_states = self.share_encoder(data)
        fea_1_sha, fea_2_sha = fea_sha[:, :256, ...], fea_sha[:, 256:, ...]
        # hidden_states1, hidden_states1 =
        ConvBlock = self.encoder1(data[:, 0:1, ...])

        f1_high = self.f1(fea_1_sha)  # [B,16*4*4*8=2048]
        f2_high = self.f2(fea_2_sha)
        f1_contra = self.g1(f1_high.view(f1_high.size(0), -1))  # [B,256] 这里用reshape/view
        f2_contra = self.g2(f2_high.view(f2_high.size(0), -1))

        f1_h = self.t1(fea_1_spe)  # [B,256,4,4,8] -> [B,16,4,4,8]
        f2_h = self.t2(fea_2_spe)
        f1_clasf = self.classifier1(f1_h.view(f1_h.size(0), -1))  # classifier(B, 2048)
        f2_clasf = self.classifier2(f2_h.view(f2_h.size(0), -1))

        # scheme1-相同模态进行concat
        fea_1_cat = torch.cat([fea_1_sha, fea_1_spe], dim=1)  # concat on channel dimension [B,512,4,4,8]
        fea_2_cat = torch.cat([fea_2_sha, fea_2_spe], dim=1)
        fea_1_proj = self.projection1(fea_1_cat)  # [B,512,4,4,8] -> [B,256,4,4,8]
        fea_2_proj = self.projection2(fea_2_cat)
        to_decoder = self.projection3(torch.cat([fea_1_proj, fea_2_proj], dim=1))  # [B,512,4,4,8]

        # scheme2-相同编码器的进行concat

        # # scheme3-share先concat，后采用feature fusion
        # share_fea = self.projection1(torch.cat([fea_1_sha, fea_2_sha], dim=1)) # 256
        # out_fea = self.fusion(share_fea, fea_1_spe, fea_2_spe)

        # only use feature map from T2 for skip connection -> since it's shared encoder, use shared feature map for skip connection
        enc1 = hidden_states[0]  # [2,32,32,32,64] -> [2,32,32,64,128]
        enc2 = hidden_states[1]  # [2,64,16,16,32] -> [2,64,16,32,64]
        enc3 = hidden_states[2]  # [2,128,8,8,16] -> [2,128,8,16,32]
        enc4 = hidden_states[3]  # [2,256,4,4,8]
        # print(to_decoder.shape) [2, 256, 4, 4, 8]
        # print(enc3.shape)

        dec3 = self.decoder5(to_decoder, enc3)  # 1-
        # dec3 = self.decoder5(out_fea, enc3) #3-
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, ConvBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            return logits, f1_contra, f2_contra, f1_clasf, f2_clasf
        else:
            logits = self.out1(out)
            return logits


class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[32 * 32 * 64, 16 * 16 * 32, 8 * 8 * 16, 4 * 4 * 8], dims=[32, 64, 128, 256],
                 proj_size=[64, 64, 64, 32], depths=[3, 3, 3, 3], num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.0, transformer_dropout_rate=0.1, hidden_size=256, **kwargs):
        super().__init__()
        self.feat_size = (4, 4, 8)
        self.hidden_size = hidden_size

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
                x = self.proj_feat(x, self.hidden_size, self.feat_size)
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UnetrPPEncoderv1(nn.Module):
    def __init__(self, input_size=[32 * 32 * 64, 16 * 16 * 32, 8 * 8 * 16, 4 * 4 * 8], dims=[32, 64, 128, 256],
                 proj_size=[64, 64, 64, 32], depths=[3, 3, 3, 3], num_heads=4, spatial_dims=3, in_channels=2,
                 dropout=0.0, transformer_dropout_rate=0.1, hidden_size=256, **kwargs):
        super().__init__()

        self.feat_size = (4, 4, 8)
        self.hidden_size = hidden_size

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # stem_layer = nn.Sequential(
        #     get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4),
        #                    dropout=dropout, conv_only=True, ),
        #     get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        # )
        # self.downsample_layers.append(stem_layer)
        # for i in range(3):
        #     downsample_layer = nn.Sequential(
        #         get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
        #                        dropout=dropout, conv_only=True, ),
        #         get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
        #     )
        #     self.downsample_layers.append(downsample_layer)

        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            if i != 2:
                downsample_layer = nn.Sequential(
                    get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                                   dropout=dropout, conv_only=True, ),
                    get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
                )
            else:
                downsample_layer = nn.Sequential(
                    get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 4, 4), stride=(2, 4, 4),
                                   dropout=dropout, conv_only=True, ),
                    get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
                )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
                x = self.proj_feat(x, self.hidden_size, self.feat_size)
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class SelfAttnNetv1(nn.Module):
        def __init__(self,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    features_per_stage,
                    kernel_sizes,
                    strides,
                    n_conv_per_stage,
                    n_conv_per_stage_decoder,
                    input_channels: int = 2,
                    out_channels: int = 4,
                    n_stages: int = 5,
                    conv_op = nn.Conv3d,
                    pool: str = 'conv',
                    feature_size: int = 16,
                    hidden_size: int = 256,
                    num_heads: int = 4,
                    norm_name: Union[Tuple, str] = "instance",
                    dropout_rate: float = 0.0,
                    depths = None,
                    dims = None,
                    do_ds = True,
                    ):

            """
            nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
            """
            super().__init__()
            if isinstance(n_conv_per_stage, int):
                n_conv_per_stage = [n_conv_per_stage] * n_stages
            if isinstance(n_conv_per_stage_decoder, int):
                n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                      f"resolution stages. here: {n_stages}. " \
                                                      f"n_conv_per_stage: {n_conv_per_stage}"
            assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                    f"as we have resolution stages. here: {n_stages} " \
                                                                    f"stages, so it should have {n_stages - 1} entries. " \
                                                                    f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
            self.do_ds = do_ds
            self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.decoder = UNetDecoder(self.encoder, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, hidden_size=hidden_size)

        def forward(self, x):
            skips = self.encoder(x)
            # print(len(skips))
            # print(skips[0].shape)
            # print(skips[-1].shape)

            return self.decoder(skips)

        def compute_conv_feature_map_size(self, input_size):
            assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                    "Give input_size=(x, y(, z))!"
            return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


class SelfAttnNetv2(nn.Module):
        def __init__(self,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    features_per_stage,
                    kernel_sizes,
                    strides,
                    n_conv_per_stage,
                    n_conv_per_stage_decoder,
                    input_channels: int = 2,
                    out_channels: int = 4,
                    n_stages: int = 5,
                    conv_op = nn.Conv3d,
                    pool: str = 'conv',
                    feature_size: int = 16,
                    hidden_size: int = 256,
                    num_heads: int = 4,
                    norm_name: Union[Tuple, str] = "instance",
                    dropout_rate: float = 0.0,
                    depths = None,
                    dims = None,
                    do_ds = True,
                    ):

            """
            nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
            """
            super().__init__()
            if isinstance(n_conv_per_stage, int):
                n_conv_per_stage = [n_conv_per_stage] * n_stages
            if isinstance(n_conv_per_stage_decoder, int):
                n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                      f"resolution stages. here: {n_stages}. " \
                                                      f"n_conv_per_stage: {n_conv_per_stage}"
            assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                    f"as we have resolution stages. here: {n_stages} " \
                                                                    f"stages, so it should have {n_stages - 1} entries. " \
                                                                    f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
            self.do_ds = do_ds
            self.encoder1 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.encoder2 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.decoder = UNetDecoder2(self.encoder1, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, hidden_size=hidden_size)

        def forward(self, x):
            skips1 = self.encoder1(x[:,0:1])
            skips2 = self.encoder2(x[:,1:2])

            return self.decoder(skips1, skips2)

        def compute_conv_feature_map_size(self, input_size):
            assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                    "Give input_size=(x, y(, z))!"
            return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


class SelfAttnNetv3(nn.Module):
        def __init__(self,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    features_per_stage,
                    kernel_sizes,
                    strides,
                    n_conv_per_stage,
                    n_conv_per_stage_decoder,
                    input_channels: int = 2,
                    out_channels: int = 4,
                    n_stages: int = 5,
                    conv_op = nn.Conv3d,
                    pool: str = 'conv',
                    feature_size: int = 16,
                    hidden_size: int = 256,
                    num_heads: int = 4,
                    norm_name: Union[Tuple, str] = "instance",
                    dropout_rate: float = 0.0,
                    depths = None,
                    dims = None,
                    do_ds = True,
                    ):

            """
            nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
            """
            super().__init__()
            if isinstance(n_conv_per_stage, int):
                n_conv_per_stage = [n_conv_per_stage] * n_stages
            if isinstance(n_conv_per_stage_decoder, int):
                n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                      f"resolution stages. here: {n_stages}. " \
                                                      f"n_conv_per_stage: {n_conv_per_stage}"
            assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                    f"as we have resolution stages. here: {n_stages} " \
                                                                    f"stages, so it should have {n_stages - 1} entries. " \
                                                                    f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
            self.do_ds = do_ds
            self.encoder1 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.encoder2 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.decoder = UNetDecoder3(self.encoder1, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, hidden_size=hidden_size)

        def forward(self, x):
            skips1 = self.encoder1(x[:,0:1])
            skips2 = self.encoder2(x[:,1:2])

            return self.decoder(skips1, skips2)

        def compute_conv_feature_map_size(self, input_size):
            assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                    "Give input_size=(x, y(, z))!"
            return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

# add self attention on each level of the decoder(skip connection)
class SelfAttnNetv4(nn.Module):
        def __init__(self,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    features_per_stage,
                    kernel_sizes,
                    strides,
                    n_conv_per_stage,
                    n_conv_per_stage_decoder,
                    input_channels: int = 2,
                    out_channels: int = 4,
                    n_stages: int = 5,
                    conv_op = nn.Conv3d,
                    pool: str = 'conv',
                    feature_size: int = 16,
                    hidden_size: int = 256,
                    num_heads: int = 4,
                    norm_name: Union[Tuple, str] = "instance",
                    dropout_rate: float = 0.0,
                    depths = None,
                    dims = None,
                    do_ds = True,
                    ):

            """
            nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
            """
            super().__init__()
            if isinstance(n_conv_per_stage, int):
                n_conv_per_stage = [n_conv_per_stage] * n_stages
            if isinstance(n_conv_per_stage_decoder, int):
                n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                      f"resolution stages. here: {n_stages}. " \
                                                      f"n_conv_per_stage: {n_conv_per_stage}"
            assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                    f"as we have resolution stages. here: {n_stages} " \
                                                                    f"stages, so it should have {n_stages - 1} entries. " \
                                                                    f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
            self.do_ds = do_ds
            self.encoder1 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.encoder2 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.decoder = UNetDecoder4(self.encoder1, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, dims=[256,256,128,64,32], input_size=[4*8*16, 8*16*32, 16*32*64, 32*64*128, 64*128*256])

        def forward(self, x):
            skips1 = self.encoder1(x[:,0:1])
            skips2 = self.encoder2(x[:,1:2])
            # print(len(skips1))

            return self.decoder(skips1, skips2)

        def compute_conv_feature_map_size(self, input_size):
            assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                    "Give input_size=(x, y(, z))!"
            return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


class FinalNetv1(nn.Module):
        def __init__(self,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    features_per_stage,
                    kernel_sizes,
                    strides,
                    n_conv_per_stage,
                    n_conv_per_stage_decoder,
                    input_channels: int = 2,
                    out_channels: int = 4,
                    n_stages: int = 5,
                    conv_op = nn.Conv3d,
                    pool: str = 'conv',
                    feature_size: int = 16,
                    hidden_size: int = 256,
                    num_heads: int = 4,
                    norm_name: Union[Tuple, str] = "instance",
                    dropout_rate: float = 0.0,
                    depths = None,
                    dims = None,
                    do_ds = True,
                    ):

            """
            nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
            """
            super().__init__()
            if isinstance(n_conv_per_stage, int):
                n_conv_per_stage = [n_conv_per_stage] * n_stages
            if isinstance(n_conv_per_stage_decoder, int):
                n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                      f"resolution stages. here: {n_stages}. " \
                                                      f"n_conv_per_stage: {n_conv_per_stage}"
            assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                    f"as we have resolution stages. here: {n_stages} " \
                                                                    f"stages, so it should have {n_stages - 1} entries. " \
                                                                    f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
            self.do_ds = do_ds
            self.encoder1 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.encoder2 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.decoder = UNetDecoder5(self.encoder1, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, hidden_size=hidden_size)
            # self.projection1 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=3, padding=1, bias=True),
            #                                  nn.Conv3d(128, 64, kernel_size=1),
            #                                  nn.BatchNorm3d(64),
            #                                  nn.ReLU(),
            #                                  nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=True),
            #                                  nn.Conv3d(32, 16, kernel_size=1),
            #                                  nn.BatchNorm3d(16),
            #                                  nn.ReLU(),
            #                                  # nn.Linear(128,64),
            #                                  # nn.BatchNorm1d(64),
            #                                  # nn.ReLU(),
            #                                  )
            # self.projection2 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=3, padding=1, bias=True),
            #                                  nn.Conv3d(128, 64, kernel_size=1),
            #                                  nn.BatchNorm3d(64),
            #                                  nn.ReLU(),
            #                                  nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=True),
            #                                  nn.Conv3d(32, 16, kernel_size=1),
            #                                  nn.BatchNorm3d(16),
            #                                  nn.ReLU(),
            #                                  # nn.Linear(128,64),
            #                                  # nn.BatchNorm1d(64),
            #                                  # nn.ReLU(),
            #                                  )

        def forward(self, x):
            skips1 = self.encoder1(x[:,0:1])
            # print(skips1[-1].shape)
            # exit()
            skips2 = self.encoder2(x[:,1:2])
            # proj1 = self.projection1(skips1[-1])
            # proj2 = self.projection2(skips2[-1])
            seg_out, distance_map = self.decoder(skips1, skips2)

            # return seg_out, distance_map, proj1, proj2
            return seg_out, distance_map

        def compute_conv_feature_map_size(self, input_size):
            assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                    "Give input_size=(x, y(, z))!"
            return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

# modified 9.14, add cross attn + self attn like in TranSiam
class FinalNetv2(nn.Module):
        def __init__(self,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    features_per_stage,
                    kernel_sizes,
                    strides,
                    n_conv_per_stage,
                    n_conv_per_stage_decoder,
                    input_channels: int = 2,
                    out_channels: int = 4,
                    n_stages: int = 5,
                    conv_op = nn.Conv3d,
                    pool: str = 'conv',
                    feature_size: int = 16,
                    hidden_size: int = 256,
                    num_heads: int = 4,
                    norm_name: Union[Tuple, str] = "instance",
                    dropout_rate: float = 0.0,
                    depths = None,
                    dims = None,
                    do_ds = True,
                    ):

            """
            nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
            """
            super().__init__()
            if isinstance(n_conv_per_stage, int):
                n_conv_per_stage = [n_conv_per_stage] * n_stages
            if isinstance(n_conv_per_stage_decoder, int):
                n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                      f"resolution stages. here: {n_stages}. " \
                                                      f"n_conv_per_stage: {n_conv_per_stage}"
            assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                    f"as we have resolution stages. here: {n_stages} " \
                                                                    f"stages, so it should have {n_stages - 1} entries. " \
                                                                    f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
            self.do_ds = do_ds
            self.encoder1 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.encoder2 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.decoder1 = UNetDecoder6(self.encoder1, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, hidden_size=hidden_size)
            self.decoder2 = UNetDecoder6(self.encoder2, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, hidden_size=hidden_size)

            self.pos_embed1 = nn.Parameter(torch.zeros(1, 128, hidden_size))
            self.pos_embed2 = nn.Parameter(torch.zeros(1, 128, hidden_size))
            self.pos_embed3 = nn.Parameter(torch.zeros(1, 128, hidden_size))
            self.pos_embed4 = nn.Parameter(torch.zeros(1, 128, hidden_size))
            self.crossattn = Cross_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            self.selfattn1 = Self_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            self.selfattn2 = Self_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            self.hidden_size = hidden_size
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.norm3 = nn.LayerNorm(hidden_size)
            self.norm4 = nn.LayerNorm(hidden_size)
            self.drop_path = DropPath(0)

        def forward(self, x):
            skips1 = self.encoder1(x[:,0:1])
            # print(skips1[-1].shape)
            # exit()
            skips2 = self.encoder2(x[:,1:2])
            # proj1 = self.projection1(skips1[-1])
            # proj2 = self.projection2(skips2[-1])
            x1 = skips1[-1]
            x2 = skips2[-1]

            B, C, H, W, D = x1.shape  # 2,256,4,4,8
            x1 = x1.reshape(B, C, H * W * D).permute(0, 2, 1)  # 2,128,256
            x2 = x2.reshape(B, C, H * W * D).permute(0, 2, 1)  # 2,128,256
            if self.pos_embed1 is not None:
                x1 = x1 + self.pos_embed1  # 2,128,256
                x2 = x2 + self.pos_embed2  # 2,128,256
            # attn1 = x1 + self.drop_path(self.attn(self.norm1(x1)))
            # attn1 = x1 + self.attn1(self.norm1(x1))
            # attn2 = x2 + self.attn2(self.norm2(x2))
            attn1, attn2 = self.crossattn(self.norm1(x2), self.norm2(x1))
            attn1 = attn1 + x2 #+ self.pos_embed3
            selfattn1 = self.selfattn1(self.norm3(attn1))
            y1 = x1 + selfattn1

            attn2 = attn2 + x1 #+ self.pos_embed4
            selfattn2 = self.selfattn2(self.norm4(attn2))
            y2 = x2 + selfattn2

            attn_skip1 = y1.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
            attn_skip2 = y2.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)

            seg_out1 = self.decoder1(skips1, attn_skip1)
            seg_out2 = self.decoder2(skips2, attn_skip2)
            if not self.do_ds:
                test_output = (seg_out1 + seg_out2) / 2
                return test_output

            return seg_out1, seg_out2

        def compute_conv_feature_map_size(self, input_size):
            assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                    "Give input_size=(x, y(, z))!"
            return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


class FinalNetv3(nn.Module):
        def __init__(self,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    features_per_stage,
                    kernel_sizes,
                    strides,
                    n_conv_per_stage,
                    n_conv_per_stage_decoder,
                    input_channels: int = 2,
                    out_channels: int = 4,
                    n_stages: int = 5,
                    conv_op = nn.Conv3d,
                    pool: str = 'conv',
                    feature_size: int = 16,
                    hidden_size: int = 256,
                    num_heads: int = 4,
                    norm_name: Union[Tuple, str] = "instance",
                    dropout_rate: float = 0.0,
                    depths = None,
                    dims = None,
                    do_ds = True,
                    ):

            """
            nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
            """
            super().__init__()
            if isinstance(n_conv_per_stage, int):
                n_conv_per_stage = [n_conv_per_stage] * n_stages
            if isinstance(n_conv_per_stage_decoder, int):
                n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                      f"resolution stages. here: {n_stages}. " \
                                                      f"n_conv_per_stage: {n_conv_per_stage}"
            assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                    f"as we have resolution stages. here: {n_stages} " \
                                                                    f"stages, so it should have {n_stages - 1} entries. " \
                                                                    f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
            self.do_ds = do_ds
            self.encoder1 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.encoder2 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.decoder1 = UNetDecoder6(self.encoder1, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, hidden_size=hidden_size)
            self.decoder2 = UNetDecoder6(self.encoder2, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, hidden_size=hidden_size)

            self.pos_embed1 = nn.Parameter(torch.zeros(1, 128, hidden_size))
            self.pos_embed2 = nn.Parameter(torch.zeros(1, 128, hidden_size))
            # self.attn1 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            # self.attn2 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            self.crossattn = Cross_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            self.selfattn1 = Self_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            self.selfattn2 = Self_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            # self.conv1_1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
            # self.conv1_2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))
            # self.conv2_1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
            # self.conv2_2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))
            self.hidden_size = hidden_size
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.drop_path = DropPath(0)
            self.proj1 = nn.Conv3d(in_channels=5, out_channels=1, kernel_size=1)
            self.proj2 = nn.Conv3d(in_channels=5, out_channels=1, kernel_size=1)

        def forward(self, x):
            skips1 = self.encoder1(x[:,0:1])
            # print(skips1[-1].shape)
            # exit()
            skips2 = self.encoder2(x[:,1:2])
            # proj1 = self.projection1(skips1[-1])
            # proj2 = self.projection2(skips2[-1])
            x1 = skips1[-1]
            x2 = skips2[-1]

            B, C, H, W, D = x1.shape  # 2,256,4,4,8
            x1 = x1.reshape(B, C, H * W * D).permute(0, 2, 1)  # 2,128,256
            x2 = x2.reshape(B, C, H * W * D).permute(0, 2, 1)  # 2,128,256
            if self.pos_embed1 is not None:
                x1 = x1 + self.pos_embed1  # 2,128,256
                x2 = x2 + self.pos_embed2  # 2,128,256
            # attn1 = x1 + self.drop_path(self.attn(self.norm1(x1)))
            # attn1 = x1 + self.attn1(self.norm1(x1))
            # attn2 = x2 + self.attn2(self.norm2(x2))
            attn1, attn2 = self.crossattn(self.norm1(x2), self.norm2(x1))
            attn1 = attn1 + x2
            selfattn1 = self.selfattn1(attn1)
            y1 = x1 + selfattn1
            attn2 = attn2 + x1
            selfattn2 = self.selfattn2(attn2)
            y2 = x2 + selfattn2

            attn_skip1 = y1.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
            attn_skip2 = y2.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)

            seg_out1 = self.decoder1(skips1, attn_skip1)
            seg_out2 = self.decoder2(skips1, attn_skip2)
            conf1 = self.proj1(seg_out1[0])
            conf2 = self.proj2(seg_out2[0])
            if not self.do_ds:
                test_output = (seg_out1 + seg_out2) / 2
                return test_output
            return seg_out1, seg_out2, conf1, conf2

        def compute_conv_feature_map_size(self, input_size):
            assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                    "Give input_size=(x, y(, z))!"
            return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

# modified 9,15, based on v2
class FinalNetv4(nn.Module):
        def __init__(self,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    features_per_stage,
                    kernel_sizes,
                    strides,
                    n_conv_per_stage,
                    n_conv_per_stage_decoder,
                    input_channels: int = 2,
                    out_channels: int = 4,
                    n_stages: int = 5,
                    conv_op = nn.Conv3d,
                    pool: str = 'conv',
                    feature_size: int = 16,
                    hidden_size: int = 256,
                    num_heads: int = 4,
                    norm_name: Union[Tuple, str] = "instance",
                    dropout_rate: float = 0.0,
                    depths = None,
                    dims = None,
                    do_ds = True,
                    ):

            """
            nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
            """
            super().__init__()
            if isinstance(n_conv_per_stage, int):
                n_conv_per_stage = [n_conv_per_stage] * n_stages
            if isinstance(n_conv_per_stage_decoder, int):
                n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                      f"resolution stages. here: {n_stages}. " \
                                                      f"n_conv_per_stage: {n_conv_per_stage}"
            assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                    f"as we have resolution stages. here: {n_stages} " \
                                                                    f"stages, so it should have {n_stages - 1} entries. " \
                                                                    f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
            self.do_ds = do_ds
            self.encoder1 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.encoder2 = PlainConvEncoder(input_channels//2, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                            nonlin_first=False)
            self.decoder1 = UNetDecoder6(self.encoder1, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, hidden_size=hidden_size)
            self.decoder2 = UNetDecoder6(self.encoder2, num_classes=out_channels, n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=self.do_ds,
                                       nonlin_first=False, hidden_size=hidden_size)

            self.pos_embed1 = nn.Parameter(torch.zeros(1, 128, hidden_size))
            self.pos_embed2 = nn.Parameter(torch.zeros(1, 128, hidden_size))
            self.crossattn = Cross_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            self.selfattn1 = Self_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            self.selfattn2 = Self_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
            self.hidden_size = hidden_size
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.drop_path = DropPath(0)

        def forward(self, x):
            skips1 = self.encoder1(x[:,0:1])
            # print(skips1[-1].shape)
            # exit()
            skips2 = self.encoder2(x[:,1:2])
            seg_out1 = self.decoder1(skips1)
            seg_out2 = self.decoder2(skips2)
            if not self.do_ds:
                test_output = (seg_out1 + seg_out2) / 2
                return test_output

            return seg_out1, seg_out2

        def compute_conv_feature_map_size(self, input_size):
            assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                    "Give input_size=(x, y(, z))!"
            return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


class Fusion_layer(nn.Module):
    def __init__(self, dim):
        super(Fusion_layer, self).__init__()
        self.dim = dim
        self.conv1 = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )
        self.proj1 = nn.Sequential(
            nn.Conv3d(dim*2, dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )
        self.proj2 = nn.Sequential(
            nn.Conv3d(dim*2, dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        if x1.size()[1] != x2.size()[1]:
            x2 = self.proj1(x2)
            x3 = self.proj2(x3)
        x1_0 = x1 * x2
        x1_0 = torch.cat((x1_0, x1), dim=1)
        x1_0 = self.conv1(x1_0)

        x2_0 = x1 * x3
        x2_0 = torch.cat((x2_0, x1), dim=1)
        x2_0 = self.conv2(x2_0)

        x = torch.cat((x1_0, x2_0), dim=1)
        x = self.conv3(x)

        x3_0 = x1 * x2 * x3
        x = torch.cat((x3_0, x), dim=1)
        x = self.conv4(x)

        return x



if __name__ == '__main__':
    unetr_pp_encoder = UnetrPPEncoder()
    input = torch.randn((2, 2, 64, 128, 256))
    x, hidden_states = unetr_pp_encoder(input)
    print(x.shape)
    print(hidden_states[-1].shape)
    a, b = x.chunk(2, 1)
    # b = x.chunk(2,0)
    print(a.shape)
    # print(x==hidden_states[-1])
    # model = HybridNet()
