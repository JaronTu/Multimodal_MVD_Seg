import torch
import einops
from torch import nn
from typing import Tuple, Union, List, Type
from timm.models.layers import trunc_normal_
from nnunetv2.training.my_network.UNetRPP.dynunet_block import UnetOutBlock, UnetResBlock
from nnunetv2.training.my_network.UNetRPP.model_components import UnetrPPEncoder, UnetrUpBlock
from nnunetv2.training.my_network.UNetRPP.dynunet_block import get_conv_layer, UnetResBlock
from nnunetv2.training.my_network.UNetRPP.transformer_block import TransformerBlock
from nnunetv2.training.my_network.UNetRPP.layers import LayerNorm
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from monai.networks.layers.utils import get_norm_layer
from monai.networks.nets import SwinUNETR


class EnhancedFeature(nn.Module):
    def __init__(self, in_chans, is_first=False):
        super().__init__()
        self.is_first = is_first
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=2 * in_chans, out_channels=in_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chans),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_chans, out_channels=in_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chans),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=3 * in_chans, out_channels=in_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chans),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=2 * in_chans, out_channels=in_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x0, x1, x2):
        w = torch.sigmoid(self.conv1(torch.cat((x1, x2), dim=1)))  # in_chans c in_chans -> in_chans -> sigmoid
        feat_x1 = torch.mul(x1, w)
        feat_x2 = torch.mul(x2, w)
        x = self.conv3(torch.cat((self.conv2(feat_x1 + feat_x2), x1, x2), dim=1))  # in_chans c in_chans c in_chans
        if not self.is_first:
            x = self.conv(torch.cat((x0, x), dim=1))
        return x


class HybridNet(nn.Module):
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
                                             dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=False,
                                             nonlin_first=False)
        self.spe_encoder2 = PlainConvEncoder(self.input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                                             strides,
                                             n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                             dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=False,
                                             nonlin_first=False)
        self.share_encoder = UnetrPPEncoder(depths=self.depths, num_heads=self.num_heads)
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
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 16,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 32,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 64,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
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

    def forward(self, data):
        fea_1_spe = self.spe_encoder1(data[:, 0:1, ...])  # get specific features for T2 [B,256,4,4,8]
        fea_2_spe = self.spe_encoder2(data[:, 1:2, ...])  # get specific features for TOF
        # fea_sha,  = self.share_encoder(data) # get shared features for T2 and TOF simultaneously
        # fea_1_sha, fea_2_sha
        fea_1_sha, hidden_states1 = self.share_encoder(data[:, 0:1, ...])
        fea_2_sha, hidden_states2 = self.share_encoder(data[:, 1:2, ...])
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
        # fea_1_cat = torch.cat([fea_1_sha, fea_1_spe], dim=1)  # concat on channel dimension [B,512,4,4,8]
        # fea_2_cat = torch.cat([fea_2_sha, fea_2_spe], dim=1)
        # fea_1_proj = self.projection1(fea_1_cat) # [B,512,4,4,8] -> [B,256,4,4,8]
        # fea_2_proj = self.projection2(fea_2_cat)
        # to_decoder = self.projection3(torch.cat([fea_1_proj, fea_2_proj], dim=1)) #[B,512,4,4,8]

        # scheme2-相同编码器的进行concat

        # scheme3-share先concat，后采用feature fusion
        share_fea = self.projection1(torch.cat([fea_1_sha, fea_2_sha], dim=1))  # 256
        out_fea = self.fusion(share_fea, fea_1_spe, fea_2_spe)

        # only use feature map from T2 for skip connection
        enc1 = hidden_states1[0]  # [2,32,32,32,64]
        enc2 = hidden_states1[1]  # [2,64,16,16,32]
        enc3 = hidden_states1[2]  # [2,128,8,8,16]
        enc4 = hidden_states1[3]  # [2,256,4,4,8]
        # print(to_decoder.shape) [2, 256, 4, 4, 8]
        # print(enc3.shape)

        # dec3 = self.decoder5(to_decoder, enc3) #1-
        dec3 = self.decoder5(out_fea, enc3)  # 3-
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, ConvBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            return logits, f1_contra, f2_contra, f1_clasf, f2_clasf
        else:
            logits = self.out1(out)
            return logits


# use conv encoder/only use information from conv decoder for skip connection
class HybridNet_v1(nn.Module):
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
        self.share_encoder = UnetrPPEncoder(depths=self.depths, num_heads=self.num_heads)
        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder = UNetDecoder(self.spe_encoder1, out_channels, n_conv_per_stage_decoder, self.do_ds,
                                   nonlin_first=False)
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

    def forward(self, data):
        skips1 = self.spe_encoder1(data[:, 0:1, ...])  # get specific features for T2 [B,256,4,4,8]
        skips2 = self.spe_encoder2(data[:, 1:2, ...])  # get specific features for TOF
        # print(len(fea_2_spe))
        fea_1_spe = skips1[-1]
        fea_2_spe = skips2[-1]
        # fea_sha,  = self.share_encoder(data) # get shared features for T2 and TOF simultaneously
        # fea_1_sha, fea_2_sha
        fea_1_sha, hidden_states1 = self.share_encoder(data[:, 0:1, ...])
        fea_2_sha, hidden_states2 = self.share_encoder(data[:, 1:2, ...])
        ConvBlock = self.encoder1(data[:, 0:1, ...])

        f1_high = self.f1(fea_1_sha)  # [B,16*4*4*8=2048]
        f2_high = self.f2(fea_2_sha)
        f1_contra = self.g1(f1_high.view(f1_high.size(0), -1))  # [B,256] 这里用reshape/view
        f2_contra = self.g2(f2_high.view(f2_high.size(0), -1))

        f1_h = self.t1(fea_1_spe)  # [B,256,4,4,8] -> [B,16,4,4,8]
        f2_h = self.t2(fea_2_spe)
        f1_clasf = self.classifier1(f1_h.view(f1_h.size(0), -1))  # classifier(B, 2048)
        f2_clasf = self.classifier2(f2_h.view(f2_h.size(0), -1))
        fea_1_cat = torch.cat([fea_1_sha, fea_1_spe], dim=1)  # concat on channel dimension [B,512,4,4,8]
        fea_2_cat = torch.cat([fea_2_sha, fea_2_spe], dim=1)
        fea_1_proj = self.projection1(fea_1_cat)  # [B,512,4,4,8] -> [B,256,4,4,8]
        fea_2_proj = self.projection2(fea_2_cat)
        to_decoder = self.projection3(torch.cat([fea_1_proj, fea_2_proj], dim=1))  # [B,512,4,4,8]
        # only use feature map from T2 for skip connection
        # enc1 = hidden_states1[0] #[2,32,32,32,64]
        # enc2 = hidden_states1[1] #[2,64,16,16,32]
        # enc3 = hidden_states1[2] #[2,128,8,8,16]
        # enc4 = hidden_states1[3] #[2,256,4,4,8]
        # # print(to_decoder.shape) [2, 256, 4, 4, 8]
        # # print(enc3.shape)
        # dec3 = self.decoder5(to_decoder, enc3)
        # dec2 = self.decoder4(dec3, enc2)
        # dec1 = self.decoder3(dec2, enc1)
        # out = self.decoder2(dec1, ConvBlock)
        output = self.decoder(skips1)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            return logits, f1_contra, f2_contra, f1_clasf, f2_clasf
        else:
            logits = self.out1(out)
            return logits


# add skip connection between shared feature and projected features
class HybridNet_v2(nn.Module):
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
                                             dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=False,
                                             nonlin_first=False)
        self.spe_encoder2 = PlainConvEncoder(self.input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                                             strides,
                                             n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                             dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=False,
                                             nonlin_first=False)
        self.share_encoder = UnetrPPEncoder(depths=self.depths, num_heads=self.num_heads)
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
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 16,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 32,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 64,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=64 * 128 * 256,
            conv_decoder=True,
        )
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

    def forward(self, data):
        fea_1_spe = self.spe_encoder1(data[:, 0:1, ...])  # get specific features for T2 [B,256,4,4,8]
        fea_2_spe = self.spe_encoder2(data[:, 1:2, ...])  # get specific features for TOF
        # fea_sha,  = self.share_encoder(data) # get shared features for T2 and TOF simultaneously
        # fea_1_sha, fea_2_sha
        fea_1_sha, hidden_states1 = self.share_encoder(data[:, 0:1, ...])
        fea_2_sha, hidden_states2 = self.share_encoder(data[:, 1:2, ...])
        ConvBlock = self.encoder1(data[:, 0:1, ...])

        f1_high = self.f1(fea_1_sha)  # [B,16*4*4*8=2048]
        f2_high = self.f2(fea_2_sha)
        f1_contra = self.g1(f1_high.view(f1_high.size(0), -1))  # [B,256] 这里用reshape/view
        f2_contra = self.g2(f2_high.view(f2_high.size(0), -1))

        f1_h = self.t1(fea_1_spe)  # [B,256,4,4,8] -> [B,16,4,4,8]
        f2_h = self.t2(fea_2_spe)
        f1_clasf = self.classifier1(f1_h.view(f1_h.size(0), -1))  # classifier(B, 2048)
        f2_clasf = self.classifier2(f2_h.view(f2_h.size(0), -1))
        fea_1_cat = torch.cat([fea_1_sha, fea_1_spe], dim=1)  # concat on channel dimension [B,512,4,4,8]
        fea_2_cat = torch.cat([fea_2_sha, fea_2_spe], dim=1)
        fea_1_proj = self.projection1(fea_1_cat)  # [B,512,4,4,8] -> [B,256,4,4,8]
        fea_2_proj = self.projection2(fea_2_cat)
        fea_1_proj += fea_1_sha
        fea_2_proj += fea_2_sha
        to_decoder = self.projection3(torch.cat([fea_1_proj, fea_2_proj], dim=1))  # [B,512,4,4,8]
        # only use feature map from T2 for skip connection
        enc1 = hidden_states1[0]  # [2,32,32,32,64]
        enc2 = hidden_states1[1]  # [2,64,16,16,32]
        enc3 = hidden_states1[2]  # [2,128,8,8,16]
        enc4 = hidden_states1[3]  # [2,256,4,4,8]
        # print(to_decoder.shape) [2, 256, 4, 4, 8]
        # print(enc3.shape)
        dec3 = self.decoder5(to_decoder, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, ConvBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            return logits, f1_contra, f2_contra, f1_clasf, f2_clasf
        else:
            logits = self.out1(out)
            return logits


class HybridNet_v3(nn.Module):
    def __init__(self,
                 out_channels: int = 4,
                 feature_size: int = 16,
                 num_heads: int = 4,
                 norm_name: Union[Tuple, str] = "instance",
                 depths=None,
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
        self.trans_encoder1 = UnetrPPEncoder(depths=self.depths, num_heads=self.num_heads)
        self.trans_encoder2 = UnetrPPEncoder(depths=self.depths, num_heads=self.num_heads)
        self.trans_encoder0 = UnetrPPEncoder_fusion(depths=self.depths, num_heads=self.num_heads)
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
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 16,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 32,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 64,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=64 * 128 * 256,
            conv_decoder=True,
        )
        self.fusion = EnhancedFeature(in_chans=256)
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)
        # self.projection1 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=3, padding=1, bias=True),
        #                                  nn.Conv3d(256, 256, kernel_size=1))
        # self.projection2 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=3, padding=1, bias=True),
        #                                  nn.Conv3d(256, 256, kernel_size=1))
        # self.projection3 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=3, padding=1, bias=True),
        #                                  nn.Conv3d(256, 256, kernel_size=1))
        # # self.projection = nn.Conv3d(512, 256, kernel_size=3, padding=1, bias=True # a simple dim reduction 512->256
        # # self.projection = nn.Conv3d(512, 256, kernel_size=1))
        # self.f1 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, padding=1, bias=True),
        #                         nn.Conv3d(64, 16, kernel_size=1))
        # self.f2 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, padding=1, bias=True),
        #                         nn.Conv3d(64, 16, kernel_size=1))
        # self.g1 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
        #                         # nn.BatchNorm1d(1024),
        #                         nn.ReLU(),
        #                         nn.Linear(in_features=1024, out_features=256))
        # self.g2 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
        #                         # nn.BatchNorm1d(1024),
        #                         nn.ReLU(),
        #                         nn.Linear(in_features=1024, out_features=256))
        # self.t1 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, padding=1, bias=True),
        #                         nn.Conv3d(64, 16, kernel_size=1))
        # self.t2 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, padding=1, bias=True),
        #                         nn.Conv3d(64, 16, kernel_size=1))
        # self.classifier1 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
        #                                 nn.BatchNorm1d(1024),
        #                                 nn.ReLU(),
        #                                 nn.Linear(in_features=1024, out_features=256),
        #                                 nn.Linear(in_features=256, out_features=100),
        #                                 nn.BatchNorm1d(100),
        #                                 nn.ReLU(),
        #                                 nn.Linear(in_features=100, out_features=1),
        #                                 # nn.Sigmoid()
        #                                 )
        # self.classifier2 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
        #                                 nn.BatchNorm1d(1024),
        #                                 nn.ReLU(),
        #                                 nn.Linear(in_features=1024, out_features=256),
        #                                 nn.Linear(in_features=256, out_features=100),
        #                                 nn.BatchNorm1d(100),
        #                                 nn.ReLU(),
        #                                 nn.Linear(in_features=100, out_features=1),
        #                                 # nn.Sigmoid()
        #                                 ) # use cross entropy lossr(data) # get shared features for T2 and TOF simultaneously [B,256,4,8,8]

    def forward(self, data):
        fea_1_spe, hidden_states1 = self.trans_encoder1(data[:, 0:1, ...])  # get specific features for T2 [B,256,4,4,8]
        fea_2_spe, hidden_states2 = self.trans_encoder2(data[:, 1:2, ...])  # get specific features for TOF
        # print(hidden_states1[0].shape)
        # print(hidden_states1[1].shape)
        x = self.trans_encoder0(hidden_states1, hidden_states2)

        # ConvBlock = self.encoder1(data[:, 0:1, ...])

        # share_fea = self.projection1(torch.cat([fea_1_sha, fea_2_sha], dim=1)) # 256
        # out_fea = self.fusion(share_fea, fea_1_spe, fea_2_spe)

        # only use feature map from T2 for skip connection
        enc1 = hidden_states1[0]  # [2,32,32,32,64]
        enc2 = hidden_states1[1]  # [2,64,16,16,32]
        enc3 = hidden_states1[2]  # [2,128,8,8,16]
        enc4 = hidden_states1[3]  # [2,256,4,4,8]
        # print(to_decoder.shape) [2, 256, 4, 4, 8]
        # print(enc3.shape)

        # dec3 = self.decoder5(to_decoder, enc3) #1-
        dec3 = self.decoder5(out_fea, enc3)  # 3-
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, x)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            return logits, f1_contra, f2_contra, f1_clasf, f2_clasf
        else:
            logits = self.out1(out)
            return logits


# only one shared encoder for unetrpp encoder, skip connection different
class HybridNet_v4(nn.Module):
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
                                             dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=False,
                                             nonlin_first=False)
        self.spe_encoder2 = PlainConvEncoder(self.input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                                             strides,
                                             n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                             dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=False,
                                             nonlin_first=False)
        self.share_encoder = UnetrPPEncoder(dims=[32, 64, 128, 512], depths=self.depths, num_heads=self.num_heads,
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
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 16,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 32,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 64,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
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

    def forward(self, data):
        fea_1_spe = self.spe_encoder1(data[:, 0:1, ...])  # get specific features for T2 [B,256,4,4,8]
        fea_2_spe = self.spe_encoder2(data[:, 1:2, ...])  # get specific features for TOF
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
        enc1 = hidden_states[0]  # [2,32,32,32,64]
        enc2 = hidden_states[1]  # [2,64,16,16,32]
        enc3 = hidden_states[2]  # [2,128,8,8,16]
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


# modified based on v4; 2 modifications: 1.Transformer feature size larger; 2.decoder fusion with conv/Trans encoder
class HybridNet_v5(nn.Module):
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
        # print(len(fea_1_spe_all))
        # print(fea_1_spe_all[0].shape) [2, 32, 64, 128, 256]
        # print(fea_1_spe_all[1].shape) [2, 64, 32, 64, 128]
        # print(fea_1_spe_all[2].shape) [2, 128, 16, 32, 64]
        # print(fea_1_spe_all[4].shape) [2, 256, 4, 8, 16]
        # print(fea_1_spe_all[5].shape) [2, 256, 4, 4, 8]
        fea_2_spe_all = self.spe_encoder2(data[:, 1:2, ...])  # get specific features for TOF
        fea_2_spe = fea_2_spe_all[-1]
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
        # print("enc3",enc3.shape) [2, 128, 8, 16, 32]
        # print(fea_1_spe_all[-3].shape) [2, 256, 8, 16, 32]
        x0 = self.fusion_layer1(enc3, fea_1_spe_all[-3], fea_1_spe_all[-3])  # 8,16,32
        dec3 = self.decoder5(to_decoder, x0)  # 1-  256->128
        # dec3 = self.decoder5(out_fea, enc3) #3-
        x0 = self.fusion_layer2(enc2, fea_1_spe_all[-4], fea_1_spe_all[-4])  # 16,32,64
        dec2 = self.decoder4(dec3, x0)
        x0 = self.fusion_layer3(enc1, fea_1_spe_all[-5], fea_1_spe_all[-5])  # 32,64,128
        dec1 = self.decoder3(dec2, x0)
        out = self.decoder2(dec1, ConvBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            return logits, f1_contra, f2_contra, f1_clasf, f2_clasf
        else:
            logits = self.out1(out)
            return logits


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
        fea_2_spe = fea_2_spe_all[-1]
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

        # if self.do_ds:
        #     logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        #     return logits, f1_contra, f2_contra, f1_clasf, f2_clasf
        # else:
        #     logits = self.out1(out)
        #     return logits

class ContrastiveNet(nn.Module):
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

        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        # self.input_channels = input_channels
        self.input_channels = 1
        self.num_classes = out_channels
        if depths is None:
            self.depths = [3, 3, 3, 3]
        self.num_heads = num_heads
        self.do_ds = do_ds
        self.share_encoder = UnetrPPEncoderv1(input_size=[32 * 64 * 128, 16 * 32 * 64, 8 * 16 * 32, 4 * 4 * 8],
                                              dims=[32, 64, 128, 512], depths=self.depths, num_heads=self.num_heads,
                                              in_channels=2,
                                              hidden_size=512)
        # self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
        #                            nonlin_first=nonlin_first)
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
        self.decoder5_ = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=8 * 16 * 32,
        )
        self.decoder4_ = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 32 * 64,
        )
        self.decoder3_ = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 64 * 128,
        )
        self.decoder2_ = UnetrUpBlock(
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
        self.out1_ = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)
            self.out2_ = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3_ = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)
        dim_in=16 #256
        feat_dim=16 #256
        self.projection_head1 = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.projection_head2 = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head1 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head2 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(self.num_classes):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(self.num_classes):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

    def forward(self, data):
        # skip1 = self.share_encoder(x)
        fea_sha, hidden_states = self.share_encoder(data)
        fea_1_sha, fea_2_sha = fea_sha[:, :256, ...], fea_sha[:, 256:, ...]
        ConvBlock = self.encoder1(data[:, 0:1, ...])

        enc1 = hidden_states[0]  # [2,32,32,32,64]
        enc2 = hidden_states[1]  # [2,64,16,16,32]
        enc3 = hidden_states[2]  # [2,128,8,8,16]
        enc4 = hidden_states[3]  # [2,256,4,4,8]
        dec3 = self.decoder5(fea_1_sha, enc3)  # 1-
        # dec3 = self.decoder5(out_fea, enc3) #3-
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, ConvBlock)

        dec3_ = self.decoder5_(fea_2_sha, enc3)  # 1-
        # dec3 = self.decoder5(out_fea, enc3) #3-
        dec2_ = self.decoder4_(dec3, enc2)
        dec1_ = self.decoder3_(dec2, enc1)
        out_ = self.decoder2_(dec1, ConvBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            logits_ = [self.out1_(out_), self.out2_(dec1_), self.out3_(dec2_)]
            return logits, logits_, out, out_
        else:
            logits = self.out1(out)
            return logits
        # return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


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


class UnetrPPEncoder_fusion(nn.Module):
    def __init__(self, input_size=[32 * 32 * 64, 16 * 16 * 32, 8 * 8 * 16, 4 * 4 * 8], dims=[32, 64, 128, 256, 256],
                 proj_size=[64, 64, 64, 32], depths=[3, 3, 3, 3], num_heads=4, spatial_dims=3, in_channels=32,
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
        for i in range(2):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(3):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        # self.fusion1 = EnhancedFeature(dims[0], is_first=True)
        # self.fusion2 = EnhancedFeature(dims[1], is_first=False)
        # self.fusion3 = EnhancedFeature(dims[2], is_first=False)
        # self.fusion4 = EnhancedFeature(dims[3], is_first=False)
        self.fusion = nn.ModuleList()
        for i in range(4):
            self.fusion.append(EnhancedFeature(dims[i], is_first=i == 0))

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

    def forward_features(self, m1, m2):
        hidden_states = []

        # print("m",m1[0].shape) [2, 32, 32, 32, 64]
        # print(self.fusion[0])
        x = self.fusion[0](0, m1[0], m2[0])
        hidden_states.append(x)

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        hidden_states.append(x)

        x = self.fusion[1](x, m1[1], m2[1])
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        hidden_states.append(x)

        x = self.fusion[2](x, m1[2], m2[2])
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        # hidden_states.append(x)

        # for i in range(1, 3):
        #     x = self.fusion[i](x, m1[i], m2[i])
        #     x = self.downsample_layers[i](x)
        #     x = self.stages[i](x)
        #     if i == 2:  # Reshape the output of the last stage
        #         x = einops.rearrange(x, "b c h w d -> b (h w d) c")
        #         x = self.proj_feat(x, self.hidden_size, self.feat_size)
        #     hidden_states.append(x)

        x = self.fusion[-1](x, m1[3], m2[3])
        hidden_states.append(x)
        return x, hidden_states

    def forward(self, m1, m2):
        x, hidden_states = self.forward_features(m1, m2)
        return x, hidden_states


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
