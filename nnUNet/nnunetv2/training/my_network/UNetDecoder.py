import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from timm.models.layers import DropPath, trunc_normal_
from nnunetv2.utilities.helpers import softmax_helper_dim1
from scipy.ndimage import distance_transform_edt

class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False,
                 hidden_size=256):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, 128, hidden_size))
        self.attn = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.conv1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv3d(hidden_size, hidden_size,1))
        self.hidden_size=hidden_size
        self.norm1 = nn.LayerNorm(hidden_size)
        self.drop_path = DropPath(0)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        x = skips[-1]

        B, C, H, W, D = x.shape # 2,256,4,4,8
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1) # 2,128,256
        if self.pos_embed is not None:
            x = x + self.pos_embed # 2,128,256
        attn = x + self.drop_path(self.attn(self.norm1(x)))
        # if self.has_mlp:
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        attn_skip = attn.reshape(B, H, W, D, C).permute(0,4,1,2,3)
        attn = self.conv1(attn_skip)
        x = attn_skip + self.conv2(attn)
        # print(x.shape)
        lres_input = x

        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

class UNetDecoder2(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False,
                 hidden_size=256):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 3 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        self.attn1 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.attn2 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.conv1_1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv1_2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv3d(hidden_size, hidden_size,1))
        self.conv2_1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv2_2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv3d(hidden_size, hidden_size,1))
        self.hidden_size=hidden_size
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.drop_path = DropPath(0)
        self.proj_conv = nn.Sequential(nn.Conv3d(hidden_size*2, hidden_size, kernel_size=3, padding=1),
                                       nn.Conv3d(hidden_size, hidden_size, kernel_size=1))

    def forward(self, skips1, skips2):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        x1 = skips1[-1]
        x2 = skips2[-1]

        B, C, H, W, D = x1.shape # 2,256,4,4,8
        x1 = x1.reshape(B, C, H * W * D).permute(0, 2, 1) # 2,128,256
        x2 = x2.reshape(B, C, H * W * D).permute(0, 2, 1) # 2,128,256
        if self.pos_embed1 is not None:
            x1 = x1 + self.pos_embed1 # 2,128,256
            x2 = x2 + self.pos_embed2 # 2,128,256
        # attn1 = x1 + self.drop_path(self.attn(self.norm1(x1)))
        attn1 = x1 + self.attn1(self.norm1(x1))
        attn2 = x2 + self.attn2(self.norm2(x2))
        # if self.has_mlp:
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        attn_skip1 = attn1.reshape(B, H, W, D, C).permute(0,4,1,2,3)
        attn_skip2 = attn2.reshape(B, H, W, D, C).permute(0,4,1,2,3)
        attn1 = self.conv1_1(attn_skip1)
        x1 = attn_skip1 + self.conv1_2(attn1)
        attn2 = self.conv2_1(attn_skip2)
        x2 = attn_skip2 + self.conv2_2(attn2)
        # print(x.shape)
        lres_input = self.proj_conv(torch.cat((x1, x2),1))

        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips1[-(s+2)], skips2[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

class UNetDecoder3(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False,
                 hidden_size=256):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 3 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        # self.attn1 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        # self.attn2 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.crossattn = Cross_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.conv1_1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv1_2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv3d(hidden_size, hidden_size,1))
        self.conv2_1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv2_2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv3d(hidden_size, hidden_size,1))
        self.hidden_size=hidden_size
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.drop_path = DropPath(0)
        self.proj_conv = nn.Sequential(nn.Conv3d(hidden_size*2, hidden_size, kernel_size=3, padding=1),
                                       nn.Conv3d(hidden_size, hidden_size, kernel_size=1))

    def forward(self, skips1, skips2):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        x1 = skips1[-1]
        x2 = skips2[-1]

        B, C, H, W, D = x1.shape # 2,256,4,4,8
        x1 = x1.reshape(B, C, H * W * D).permute(0, 2, 1) # 2,128,256
        x2 = x2.reshape(B, C, H * W * D).permute(0, 2, 1) # 2,128,256
        if self.pos_embed1 is not None:
            x1 = x1 + self.pos_embed1 # 2,128,256
            x2 = x2 + self.pos_embed2 # 2,128,256
        # attn1 = x1 + self.drop_path(self.attn(self.norm1(x1)))
        # attn1 = x1 + self.attn1(self.norm1(x1))
        # attn2 = x2 + self.attn2(self.norm2(x2))
        attn1, attn2 = self.crossattn(self.norm1(x1), self.norm2(x2))
        attn1 = attn1 + x1
        attn2 = attn2 + x2
        # if self.has_mlp:
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        attn_skip1 = attn1.reshape(B, H, W, D, C).permute(0,4,1,2,3)
        attn_skip2 = attn2.reshape(B, H, W, D, C).permute(0,4,1,2,3)
        attn1 = self.conv1_1(attn_skip1)
        x1 = attn_skip1 + self.conv1_2(attn1)
        attn2 = self.conv2_1(attn_skip2)
        x2 = attn_skip2 + self.conv2_2(attn2)
        # print(x.shape)
        lres_input = self.proj_conv(torch.cat((x1, x2),1))

        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips1[-(s+2)], skips2[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


class UNetDecoder4(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False,
                 dims=[256,256,128,64,32], input_size=[4*8*16, 8*16*32, 16*32*64, 32*64*128, 64*128*256],
                 num_heads=[4,2,1,1,1]):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            if s<3:
                stages.append(StackedConvBlocks(
                    n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                    encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
                ))
            else:
                stages.append(StackedConvBlocks(
                    n_conv_per_stage[s - 1], encoder.conv_op, 3 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                    encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
                ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        # self.hidden_size=hidden_size
        # self.norm1 = nn.LayerNorm(hidden_size)
        # self.norm2 = nn.LayerNorm(hidden_size)
        # self.drop_path = DropPath(0)
        self.proj_conv = nn.Sequential(nn.Conv3d(256*2, 256, kernel_size=3, padding=1),
                                       nn.Conv3d(256, 256, kernel_size=1))
        # self.patch_embedding = PatchEmbedding(feature_size=128, hidden_size=hidden_size)
        # self.merge_conv = nn.Sequential(nn.Conv3d(hidden_size*2, hidden_size, kernel_size=3, padding=1),
        #                                nn.Conv3d(hidden_size, hidden_size, kernel_size=1))
        self.AttnBlock = SelfAttnBlock(input_size=128, hidden_size=256, num_heads=8)
        self.Attnstages = nn.ModuleList()
        self.merge_conv = nn.ModuleList()
        for i in range(len(self.stages)):
            # stages_block = []
            # stages_block.append(self.SelfAttnBlock(input_size, hidden_size, num_heads))
            self.Attnstages.append(SelfAttnBlock(input_size=input_size[i], hidden_size=dims[i], num_heads=num_heads[i]))
            self.merge_conv.append(nn.Sequential(nn.Conv3d(dims[i]*2, dims[i], kernel_size=3, padding=1),
                                                 nn.Conv3d(dims[i], dims[i], kernel_size=1)))

    def forward(self, skips1, skips2):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        x1 = skips1[-1]
        x2 = skips2[-1]
        # for sk in skips2:
        #     print(sk.shape)

        x1, x2 = self.AttnBlock(x1, x2)

        # print(x.shape)
        lres_input = self.proj_conv(torch.cat((x1, x2),1))

        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            # print(x.shape)
            # exit()
            if s<2:
                x1, x2 = self.Attnstages[s](skips1[-(s+2)], skips2[-(s+2)])
                skips_attn = self.merge_conv[s](torch.cat((x1, x2), 1))
                #
                # skips_attn = f"self.crossattn{s}"(skips1[-(s+2)], skips2[-(s+2)])

                x = torch.cat((x, skips_attn), 1)
                x = self.stages[s](x)
            else:
                x = torch.cat((x, skips1[-(s+2)], skips2[-(s+2)]), 1)
                x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


# modified: 9.12

class UNetDecoder5(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False,
                 hidden_size=256):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 3 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        # self.attn1 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        # self.attn2 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.crossattn = Cross_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.conv1_1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv1_2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv3d(hidden_size, hidden_size,1))
        self.conv2_1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv2_2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv3d(hidden_size, hidden_size,1))
        self.hidden_size=hidden_size
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.drop_path = DropPath(0)
        self.proj_conv = nn.Sequential(nn.Conv3d(hidden_size*2, hidden_size, kernel_size=3, padding=1),
                                       nn.Conv3d(hidden_size, hidden_size, kernel_size=1))

    def forward(self, skips1, skips2):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        x1 = skips1[-1]
        x2 = skips2[-1]

        B, C, H, W, D = x1.shape # 2,256,4,4,8
        x1 = x1.reshape(B, C, H * W * D).permute(0, 2, 1) # 2,128,256
        x2 = x2.reshape(B, C, H * W * D).permute(0, 2, 1) # 2,128,256
        if self.pos_embed1 is not None:
            x1 = x1 + self.pos_embed1 # 2,128,256
            x2 = x2 + self.pos_embed2 # 2,128,256
        # attn1 = x1 + self.drop_path(self.attn(self.norm1(x1)))
        # attn1 = x1 + self.attn1(self.norm1(x1))
        # attn2 = x2 + self.attn2(self.norm2(x2))
        attn1, attn2 = self.crossattn(self.norm1(x1), self.norm2(x2))
        attn1 = attn1 + x1
        attn2 = attn2 + x2
        # if self.has_mlp:
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        attn_skip1 = attn1.reshape(B, H, W, D, C).permute(0,4,1,2,3)
        attn_skip2 = attn2.reshape(B, H, W, D, C).permute(0,4,1,2,3)
        attn1 = self.conv1_1(attn_skip1)
        x1 = attn_skip1 + self.conv1_2(attn1)
        attn2 = self.conv2_1(attn_skip2)
        x2 = attn_skip2 + self.conv2_2(attn2)
        # print(x.shape)
        lres_input = self.proj_conv(torch.cat((x1, x2),1))

        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips1[-(s+2)], skips2[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]
        output_seg = seg_outputs[0].argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(seg_outputs[0].shape, device=x1.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        distance_map = distance_transform_edt(predicted_segmentation_onehot.cpu().numpy())
        # print(distance_map.shape)
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r, distance_map

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


#modified: 9.14
class UNetDecoder6(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False,
                 hidden_size=256):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        # self.attn1 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        # self.attn2 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.crossattn = Cross_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.selfattn = Self_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.hidden_size=hidden_size
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.drop_path = DropPath(0)

    def forward(self, skips, attn_skip):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](attn_skip)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            attn_skip = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


class UNetDecoder_return_last_fea(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False,
                 hidden_size=256):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, 128, hidden_size))
        # self.attn1 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        # self.attn2 = Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.crossattn = Cross_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.selfattn = Self_Attention(dim=hidden_size, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)
        self.hidden_size=hidden_size
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.drop_path = DropPath(0)

    def forward(self, skips, attn_skip):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](attn_skip)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if s == (len(self.stages) - 1):
                seg_fea = x
                print("seg_fea.shape", seg_fea.shape)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            attn_skip = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs, seg_fea
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=9, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads #hiddensize//num_h 128//8
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape #C-hidden_size
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale # self-attention
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, C) #attn weights 得到 features
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=9, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads #hiddensize//num_h 128//8
        self.scale = head_dim ** -0.5

        self.qkv1 = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop1 = nn.Dropout(attn_drop)
        self.attn_drop2 = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        # 用q1的是x1
        B, N, C = x1.shape #C-hidden_size
        qkv1 = self.qkv1(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        qkv2 = self.qkv2(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale # self-attention
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop1(attn1)

        x1 = (attn1 @ v2).transpose(1,2).reshape(B, N, C) #attn weights 得到 features
        x1 = self.proj1(x1)
        x1 = self.proj_drop1(x1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale # self-attention
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop2(attn2)

        x2 = (attn2 @ v1).transpose(1,2).reshape(B, N, C) #attn weights 得到 features
        x2 = self.proj2(x2)
        x2 = self.proj_drop2(x2)
        return x1, x2


class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=9, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads #hiddensize//num_h 128//8
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape #C-hidden_size
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, heads, N, C//heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale # self-attention
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) #attn weights 得到 features
        x1 = self.proj(x)
        x1 = self.proj_drop(x)

        return x


class CrossWindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads #
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),num_heads))
            # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        B_, N, C = x.shape
        q, k, v = self.query(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3), self.key(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3),self.value(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1) #

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
