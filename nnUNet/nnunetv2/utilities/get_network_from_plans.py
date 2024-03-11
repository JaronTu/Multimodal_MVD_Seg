from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.training.my_network.my_network.HybridNetwork import HybridNet, HybridNet_v4, HybridNet_v5, HybridNet_v6, ContrastiveNet
from nnunetv2.training.my_network.selfattnNet import SelfAttnNetv1, SelfAttnNetv2, SelfAttnNetv3, SelfAttnNetv4, FinalNetv1, FinalNetv2, FinalNetv3, FinalNetv4
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.training.my_network.CSNet import CSNet3D
from nnunetv2.training.my_network.WingsNet import WingsNet
from thop import profile
from ptflops import get_model_complexity_info
import torch

def get_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = configuration_manager.UNet_class_name
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accomodate that.'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    # network class name!!
    # print("label_manager",label_manager)
    num_cls = label_manager.num_segmentation_heads

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        # num_classes=num_cls*8,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    print('Number of network parameters:', sum(param.numel() for param in model.parameters()))
    # inputs = torch.randn(2,2,64,128,256)
    # print(model)
    # flops, params = profile(self, model=model, inputs=inputs)
    # print("flops:", flops, "params:", params)
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model

def get_dual_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    # segmentation_network_class_name = configuration_manager.UNet_class_name
    # mapping = {
    #     'PlainConvUNet': PlainConvUNet,
    #     'ResidualEncoderUNet': ResidualEncoderUNet
    # }
    # kwargs = {
    #     'PlainConvUNet': {
    #         'conv_bias': True,
    #         'norm_op': get_matching_instancenorm(conv_op),
    #         'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
    #         'dropout_op': None, 'dropout_op_kwargs': None,
    #         'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
    #     },
    #     'ResidualEncoderUNet': {
    #         'conv_bias': True,
    #         'norm_op': get_matching_instancenorm(conv_op),
    #         'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
    #         'dropout_op': None, 'dropout_op_kwargs': None,
    #         'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
    #     }
    # }
    # assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
    #                                                           'is non-standard (maybe your own?). Yo\'ll have to dive ' \
    #                                                           'into either this ' \
    #                                                           'function (get_network_from_plans) or ' \
    #                                                           'the init of your nnUNetModule to accomodate that.'
    # network_class = mapping[segmentation_network_class_name]

    # conv_or_blocks_per_stage = {
    #     'n_conv_per_stage'
    #     if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
    #     'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    # }
    # network class name!!
    # print("label_manager",label_manager)
    # num_cls = label_manager.num_segmentation_heads

    # model = HybridNet_v4(
    #     conv_bias=True,
    #     norm_op = nn.InstanceNorm3d,
    #     norm_op_kwargs={'eps':1e-5, 'affine':True},
    #     dropout_op=None,
    #     dropout_op_kwargs=None,
    #     nonlin=nn.LeakyReLU,
    #     nonlin_kwargs={'inplace':True},
    #     features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 **i,
    #                         configuration_manager.unet_max_num_features) for i in range(num_stages)],
    #     kernel_sizes=configuration_manager.conv_kernel_sizes,
    #     strides = configuration_manager.pool_op_kernel_sizes,
    #     n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
    #     n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
    #     input_channels=num_input_channels,
    #     out_channels=label_manager.num_segmentation_heads,
    #     # num_classes=label_manager.num_segmentation_heads*8,
    #     n_stages=num_stages,
    #     conv_op=nn.Conv3d,
    #     feature_size=16,
    #     num_heads=4,
    #     norm_name="instance",
    #     dropout_rate=0.0,
    #     depths=None,
    #     dims=None,
    #     do_ds=deep_supervision,
    # )
    # model = HybridNet_v6(
    model = ContrastiveNet(
        conv_bias=True,
        norm_op = nn.InstanceNorm3d,
        norm_op_kwargs={'eps':1e-5, 'affine':True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace':True},
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                            configuration_manager.unet_max_num_features) for i in range(num_stages)],
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides = configuration_manager.pool_op_kernel_sizes,
        n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
        n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
        input_channels=num_input_channels,
        out_channels=label_manager.num_segmentation_heads,
        # num_classes=label_manager.num_segmentation_heads*8,
        n_stages=num_stages,
        conv_op=nn.Conv3d,
        feature_size=16,
        num_heads=4,
        norm_name="instance",
        dropout_rate=0.0,
        depths=None,
        dims=None,
        do_ds=deep_supervision,
    )
    # model = SelfAttnNetv1(
    # model = SelfAttnNetv2(
    # model = SelfAttnNetv3(
    # model = SelfAttnNetv4(
    # model = FinalNetv1(
    # model = FinalNetv2(
    # model = FinalNetv3(
    # model = FinalNetv4(
    #     conv_bias=True,
    #     norm_op = nn.InstanceNorm3d,
    #     norm_op_kwargs={'eps':1e-5, 'affine':True},
    #     dropout_op=None,
    #     dropout_op_kwargs=None,
    #     nonlin=nn.LeakyReLU,
    #     nonlin_kwargs={'inplace':True},
    #     features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
    #                         configuration_manager.unet_max_num_features) for i in range(num_stages)],
    #     kernel_sizes=configuration_manager.conv_kernel_sizes,
    #     strides = configuration_manager.pool_op_kernel_sizes,
    #     n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
    #     n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
    #     input_channels=num_input_channels,
    #     out_channels=label_manager.num_segmentation_heads,
    #     # num_classes=label_manager.num_segmentation_heads*8,
    #     n_stages=num_stages,
    #     conv_op=nn.Conv3d,
    #     feature_size=16,
    #     num_heads=4,
    #     norm_name="instance",
    #     dropout_rate=0.0,
    #     depths=None,
    #     dims=None,
    #     do_ds=deep_supervision,
    # )
    model.apply(InitWeights_He(1e-2))
    print('Number of network parameters:', sum(param.numel() for param in model.parameters()))
    # if network_class == ResidualEncoderUNet:
    #     model.apply(init_last_bn_before_add_to_0)
    return model

def get_CSNet_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    label_manager = plans_manager.get_label_manager(dataset_json)
    model = CSNet3D(classes=label_manager.num_segmentation_heads, channels=num_input_channels)
    # model.apply(InitWeights_He(1e-2))
    return model

def get_wingsnet_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    label_manager = plans_manager.get_label_manager(dataset_json)
    model = WingsNet(in_channel=num_input_channels, n_classes=label_manager.num_segmentation_heads)
    model.apply(InitWeights_He(1e-2))
    return model