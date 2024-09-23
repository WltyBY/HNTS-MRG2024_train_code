import torch
import torch.nn as nn
import yaml

from nnunetv2.training.nnUNetTrainer.DFUNet.baseblock_CNN import ResidualBlock, ConvBlock
from nnunetv2.training.nnUNetTrainer.DFUNet.pooling import ConvDownPool, ConvUpPool
from nnunetv2.training.nnUNetTrainer.DFUNet.activate_function import ACTIVATE_LAYER


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_out_channels,
        pool_out_channels,
        dropout_p,
        num_conv,
        kernel_size,
        down_scale_factor,
        normalization,
        activate,
        bias=False,
    ):
        super(DownBlock, self).__init__()

        self.conv = ResidualBlock(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            dropout_p=dropout_p,
            dim=len(kernel_size),
            num_conv=num_conv,
            kernel_size=kernel_size,
            padding_size=[(p - 1) // 2 for p in kernel_size],
            normalization=normalization,
            activate=activate,
            bias=bias,
        )
        self.downpool = ConvDownPool(
            in_channels=conv_out_channels,
            out_channels=pool_out_channels,
            dim=len(kernel_size),
            pool_kernel_size=down_scale_factor,
            normalization=normalization,
            activate=activate,
            bias=bias,
        )

    def forward(self, inputs):
        skip_features = self.conv(inputs)
        outputs = self.downpool(skip_features)
        return skip_features, outputs


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        upsample_out_channels,
        conv_out_channels,
        dropout_p,
        num_conv,
        kernel_size,
        up_scale_factor,
        normalization,
        activate,
        bias=False,
    ):
        super(UpBlock, self).__init__()
        self.uppool = ConvUpPool(
            in_channels=in_channels,
            out_channels=upsample_out_channels,
            dim=len(kernel_size),
            pool_kernel_size=up_scale_factor,
            normalization=normalization,
            activate=activate,
            bias=bias,
        )
        self.conv = ConvBlock(
            in_channels=upsample_out_channels + conv_out_channels,
            out_channels=conv_out_channels,
            dropout_p=dropout_p,
            dim=len(kernel_size),
            num_conv=num_conv,
            kernel_size=kernel_size,
            padding_size=[(p - 1) // 2 for p in kernel_size],
            normalization=normalization,
            activate=activate,
            bias=bias,
        )
        Activate_layer = ACTIVATE_LAYER[activate.lower()]
        self.activate_layer = Activate_layer()

    def forward(self, inputs, skip_features):
        up_features = self.uppool(inputs)
        outputs = self.conv(torch.cat([skip_features, up_features], 1)) + up_features
        return self.activate_layer(outputs)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_channels = self.params["in_channels"]
        features = self.params["features"]
        self.dropout_p = self.params["dropout_p"]
        self.num_conv_per_stage = self.params["num_conv_per_stage"]
        self.kernel_size = self.params["kernel_size"]
        self.pool_kernel_size = self.params["pool_kernel_size"]
        self.normalization = self.params["normalization"]
        self.activate = self.params["activate"]
        self.bias = self.params["need_bias"]

        self.Encoder_layers = nn.ModuleList()
        for i in range(len(self.num_conv_per_stage) - 1):
            self.Encoder_layers.append(
                DownBlock(
                    in_channels=self.in_channels if i == 0 else features[i],
                    conv_out_channels=features[i],
                    pool_out_channels=features[i + 1],
                    dropout_p=self.dropout_p[i],
                    num_conv=self.num_conv_per_stage[i],
                    kernel_size=self.kernel_size[i],
                    down_scale_factor=self.pool_kernel_size[i],
                    normalization=self.normalization,
                    activate=self.activate,
                    bias=self.bias,
                )
            )

        self.bottleneck = ResidualBlock(
            in_channels=features[-1],
            out_channels=features[-1],
            dropout_p=self.dropout_p[-1],
            dim=len(self.kernel_size[0]),
            num_conv=self.num_conv_per_stage[-1],
            kernel_size=self.kernel_size[-1],
            padding_size=[(p - 1) // 2 for p in self.kernel_size[-1]],
            normalization=self.normalization,
            activate=self.activate,
            bias=self.bias,
        )

    def forward(self, inputs):
        encoder_out = []
        for i in range(len(self.Encoder_layers)):
            skip_features, inputs = self.Encoder_layers[i](inputs)
            encoder_out.append(skip_features)
        encoder_out.append(self.bottleneck(inputs))
        return encoder_out


class Decoder(nn.Module):
    def __init__(self, params, output_features=False):
        super(Decoder, self).__init__()
        self.params = params
        features = self.params["features"]
        self.kernel_size = self.params["kernel_size"]
        self.pool_kernel_size = self.params["pool_kernel_size"]
        self.dropout_p = self.params["dropout_p"]
        self.num_conv_per_stage = self.params["num_conv_per_stage"]
        self.normalization = self.params["normalization"]
        self.activate = self.params["activate"]
        self.bias = self.params["need_bias"]

        self.output_features = output_features

        self.Decoder_layers = nn.ModuleList()
        for i in range(len(features) - 1):
            self.Decoder_layers.append(
                UpBlock(
                    in_channels=features[i],
                    upsample_out_channels=features[i + 1],
                    conv_out_channels=features[i + 1],
                    dropout_p=self.dropout_p[i],
                    num_conv=self.num_conv_per_stage[i],
                    kernel_size=self.kernel_size[i],
                    up_scale_factor=self.pool_kernel_size[i],
                    normalization=self.normalization,
                    activate=self.activate,
                    bias=self.bias,
                )
            )

    def forward(self, encoder_out):
        # low-resolution to high-resolution
        encoder_out = encoder_out[::-1]
        if self.output_features:
            decoder_out = []
            for i, layer in enumerate(self.Decoder_layers):
                outputs = layer(
                    encoder_out[i] if i == 0 else outputs, encoder_out[i + 1]
                )
                decoder_out.append(outputs)
        else:
            for i, layer in enumerate(self.Decoder_layers):
                decoder_out = layer(
                    encoder_out[i] if i == 0 else decoder_out, encoder_out[i + 1]
                )

        return decoder_out


class CrossAttentionModule(nn.Module):
    def __init__(self, in_dim, dim):
        super(CrossAttentionModule, self).__init__()
        if dim == 2:
            ConvLayer = nn.Conv2d
            PoolLayer = nn.AdaptiveAvgPool2d
        elif dim == 3:
            ConvLayer = nn.Conv3d
            PoolLayer = nn.AdaptiveAvgPool3d
        else:
            raise ValueError("Only support 2/3d data")

        self.spatial_wise_conv = ConvLayer(in_dim, 1, 1, bias=True)
        self.channel_wise_conv = ConvLayer(in_dim, in_dim, 1, bias=True)
        self.avg_pool = PoolLayer(1)

    def forward(self, x, context):
        """
        x: [b, c, z, y, x]
        context: [b, c, z, y, x]
        """
        # spatial attention
        context_feat_map = self.spatial_wise_conv(context)
        context_feat_map = context_feat_map.sigmoid()
        spatial_attentioned_x_feat = context_feat_map * x

        # channel-wise attention
        feat_vec = self.avg_pool(spatial_attentioned_x_feat)
        feat_vec = self.channel_wise_conv(feat_vec)
        feat_vec = feat_vec.softmax(dim=1) * feat_vec.shape[1]
        channel_attentioned_x_feat = spatial_attentioned_x_feat * feat_vec

        final_feat = channel_attentioned_x_feat + x
        return final_feat
    

class VNet(nn.Module):
    def __init__(self, params):
        super(VNet, self).__init__()
        self.need_features = params["need_features"]
        self.deep_supervision = params["deep_supervision"]

        self.encoder_params, self.decoder_params = self.get_EnDecoder_params(params)
        self.encoder = Encoder(self.encoder_params)
        self.decoder = Decoder(
            self.decoder_params,
            output_features=self.deep_supervision or self.need_features,
        )

        if len(params["kernel_size"][0]) == 2:
            Conv_layer = nn.Conv2d
        elif len(params["kernel_size"][0]) == 3:
            Conv_layer = nn.Conv3d

        if self.deep_supervision:
            self.prediction_head = nn.ModuleList()
            # we will not do deep supervision on the prediction of bottleneck output feature
            # the prediction_heads are from low to high resolution.
            for i in range(1, len(self.encoder_params["num_conv_per_stage"])):
                self.prediction_head.append(
                    Conv_layer(
                        self.decoder_params["features"][i],
                        params["out_channels"],
                        kernel_size=1,
                    )
                )
        else:
            self.prediction_head = Conv_layer(
                self.decoder_params["features"][-1],
                params["out_channels"],
                kernel_size=1,
            )

    def forward(self, inputs):
        encoder_out = self.encoder(inputs)
        decoder_out = self.decoder(encoder_out)
        if self.deep_supervision:
            outputs = []
            for i in range(len(decoder_out)):
                outputs.append(self.prediction_head[i](decoder_out[i]))
            # we assume that the multi-level prediction ranking ranges from high resolution to low resolution
            if self.need_features:
                net_out = {"feature": encoder_out + decoder_out, "pred": outputs[::-1]}
            else:
                net_out = {"pred": outputs[::-1]}
        else:
            if self.need_features:
                outputs = self.prediction_head(decoder_out[-1])
                net_out = {"feature": encoder_out + decoder_out, "pred": outputs}
            else:
                net_out = {"pred": self.prediction_head(decoder_out)}

        return net_out

    def get_EnDecoder_params(self, params):
        encoder_params = {}
        decoder_params = {}

        encoder_params["in_channels"] = params["in_channels"]
        encoder_params["features"] = params["features"]
        encoder_params["dropout_p"] = params["dropout_p"]
        encoder_params["num_conv_per_stage"] = params["num_conv_per_stage"]
        encoder_params["kernel_size"] = params["kernel_size"]
        encoder_params["pool_kernel_size"] = params["pool_kernel_size"]
        encoder_params["normalization"] = params["normalization"]
        encoder_params["activate"] = params["activate"]
        encoder_params["need_bias"] = params["need_bias"]

        assert (
            len(encoder_params["features"])
            == len(encoder_params["dropout_p"])
            == len(encoder_params["num_conv_per_stage"])
            == len(encoder_params["kernel_size"])
            == (len(encoder_params["pool_kernel_size"]) + 1)
        )

        decoder_params["features"] = params["features"][::-1]
        decoder_params["kernel_size"] = params["kernel_size"][::-1]
        decoder_params["pool_kernel_size"] = params["pool_kernel_size"][::-1]
        decoder_params["dropout_p"] = params["dropout_p"][::-1]
        decoder_params["num_conv_per_stage"] = params["num_conv_per_stage"][::-1]
        decoder_params["normalization"] = params["normalization"]
        decoder_params["activate"] = params["activate"]
        decoder_params["out_channels"] = params["out_channels"]
        decoder_params["need_bias"] = params["need_bias"]

        return encoder_params, decoder_params


class DFUNet(nn.Module):
    # Dual Flow UNet
    def __init__(self, params):
        super(DFUNet, self).__init__()
        self.need_features = params["need_features"]
        self.deep_supervision = params["deep_supervision"]
        self.context_split = params["context_split"]

        self.encoder_params, self.decoder_params = self.get_EnDecoder_params(params)
        self.encoder = Encoder(self.encoder_params)
        self.decoder = Decoder(
            self.decoder_params,
            output_features=self.deep_supervision or self.need_features,
        )

        self.context_encoder = nn.ModuleList()
        self.crossattn_lst = nn.ModuleList()
        for i in range(len(self.encoder_params["num_conv_per_stage"]) - 1):
            self.context_encoder.append(
                DownBlock(
                    in_channels=(
                        params["context_in_channels"]
                        if i == 0
                        else self.encoder_params["features"][i]
                    ),
                    conv_out_channels=self.encoder_params["features"][i],
                    pool_out_channels=self.encoder_params["features"][i + 1],
                    dropout_p=self.encoder_params["dropout_p"][i],
                    num_conv=self.encoder_params["num_conv_per_stage"][i],
                    kernel_size=self.encoder_params["kernel_size"][i],
                    down_scale_factor=self.encoder_params["pool_kernel_size"][i],
                    normalization=self.encoder_params["normalization"],
                    activate=self.encoder_params["activate"],
                    bias=self.encoder_params["need_bias"],
                )
            )
            self.crossattn_lst.append(
                CrossAttentionModule(
                    in_dim=self.encoder_params["features"][i + 1],
                    dim=len(self.encoder_params["kernel_size"][0]),
                )
            )
        assert (
            len(self.encoder.Encoder_layers)
            == len(self.context_encoder)
            == len(self.crossattn_lst)
        )

        if len(params["kernel_size"][0]) == 2:
            Conv_layer = nn.Conv2d
        elif len(params["kernel_size"][0]) == 3:
            Conv_layer = nn.Conv3d

        if self.deep_supervision:
            self.prediction_head = nn.ModuleList()
            # we will not do deep supervision on the prediction of bottleneck output feature
            # the prediction_heads are from low to high resolution.
            for i in range(1, len(self.encoder_params["num_conv_per_stage"])):
                self.prediction_head.append(
                    Conv_layer(
                        self.decoder_params["features"][i],
                        params["out_channels"],
                        kernel_size=1,
                    )
                )
        else:
            self.prediction_head = Conv_layer(
                self.decoder_params["features"][-1],
                params["out_channels"],
                kernel_size=1,
            )

    def forward(self, data):
        assert data.shape[1] > self.context_split
        x = data[:, 0:self.context_split]
        context = data[:, self.context_split:]
        encoder_output = []
        for x_encoder, context_encoder, crossattn in zip(
            self.encoder.Encoder_layers, self.context_encoder, self.crossattn_lst
        ):
            skip_feature, x = x_encoder(x)
            encoder_output.append(skip_feature)
            _, context = context_encoder(context)
            x = crossattn(x, context)
        encoder_output.append(self.encoder.bottleneck(x))

        decoder_out = self.decoder(encoder_output)
        if self.deep_supervision:
            outputs = []
            for i in range(len(decoder_out)):
                outputs.append(self.prediction_head[i](decoder_out[i]))
            # we assume that the multi-level prediction ranking ranges from high resolution to low resolution
            if self.need_features:
                net_out = {
                    "feature": encoder_output + decoder_out,
                    "pred": outputs[::-1],
                }
            else:
                net_out = {"pred": outputs[::-1]}
        else:
            if self.need_features:
                outputs = self.prediction_head(decoder_out[-1])
                net_out = {"feature": encoder_output + decoder_out, "pred": outputs}
            else:
                net_out = {"pred": self.prediction_head(decoder_out)}

        return net_out

    def get_EnDecoder_params(self, params):
        encoder_params = {}
        decoder_params = {}

        encoder_params["in_channels"] = params["in_channels"]
        encoder_params["features"] = params["features"]
        encoder_params["dropout_p"] = params["dropout_p"]
        encoder_params["num_conv_per_stage"] = params["num_conv_per_stage"]
        encoder_params["kernel_size"] = params["kernel_size"]
        encoder_params["pool_kernel_size"] = params["pool_kernel_size"]
        encoder_params["normalization"] = params["normalization"]
        encoder_params["activate"] = params["activate"]
        encoder_params["need_bias"] = params["need_bias"]

        assert (
            len(encoder_params["features"])
            == len(encoder_params["dropout_p"])
            == len(encoder_params["num_conv_per_stage"])
            == len(encoder_params["kernel_size"])
            == (len(encoder_params["pool_kernel_size"]) + 1)
        )

        decoder_params["features"] = params["features"][::-1]
        decoder_params["kernel_size"] = params["kernel_size"][::-1]
        decoder_params["pool_kernel_size"] = params["pool_kernel_size"][::-1]
        decoder_params["dropout_p"] = params["dropout_p"][::-1]
        decoder_params["num_conv_per_stage"] = params["num_conv_per_stage"][::-1]
        decoder_params["normalization"] = params["normalization"]
        decoder_params["activate"] = params["activate"]
        decoder_params["out_channels"] = params["out_channels"]
        decoder_params["need_bias"] = params["need_bias"]

        return encoder_params, decoder_params


def open_yaml(file_path, mode="r"):
    with open(file_path, mode) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


if __name__ == "__main__":
    import os
    import time

    data = open_yaml("./wcode/net/CNN/DFUNet/DFUNet_test.yaml")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("-----VNet2d-----")
    dfunet2d = DFUNet(data["Network2d"]).to(device)
    begin = time.time()
    with torch.no_grad():
        inputs = torch.rand((16, 1, 256, 256)).to(device)
        context = torch.rand((16, 2, 256, 256)).to(device)
        outputs = dfunet2d(inputs, context)
    print("Time:", time.time() - begin)
    print("Outputs features:")
    if isinstance(outputs["feature"], (list, tuple)):
        for output in outputs["feature"]:
            print(output.shape)
    else:
        print(outputs["feature"].shape)

    print("Outputs prediction:")
    if isinstance(outputs["pred"], (list, tuple)):
        for output in outputs["pred"]:
            print(output.shape)
    else:
        print(outputs["pred"].shape)
    total = sum(p.numel() for p in dfunet2d.parameters())
    print("Total params: %.3fM" % (total / 1e6))

    print("-----VNet3d-----")
    dfunet3d = DFUNet(data["Network3d"]).to(device)
    begin = time.time()
    with torch.no_grad():
        inputs = torch.rand((2, 1, 56, 224, 160)).to(device)
        context = torch.rand((2, 2, 56, 224, 160)).to(device)
        outputs = dfunet3d(inputs, context)
    print("Time:", time.time() - begin)
    print("Outputs features:")
    if isinstance(outputs["feature"], (list, tuple)):
        for output in outputs["feature"]:
            print(output.shape)
    else:
        print(outputs["feature"].shape)

    print("Outputs prediction:")
    if isinstance(outputs["pred"], (list, tuple)):
        for output in outputs["pred"]:
            print(output.shape)
    else:
        print(outputs["pred"].shape)
    total = sum(p.numel() for p in dfunet3d.parameters())
    print("Total params: %.3fM" % (total / 1e6))
