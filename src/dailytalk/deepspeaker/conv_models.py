import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class IdentityBlock(nn.Module):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.clamp(out, 0.0, 20.0)

        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.clamp(out, 0.0, 20.0)

        out = out + residual
        out = torch.clamp(out, 0.0, 20.0)
        return out


class ConvAndResBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(out_filters)
        self.res_blocks = nn.Sequential(
            IdentityBlock(out_filters),
            IdentityBlock(out_filters),
            IdentityBlock(out_filters)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.clamp(out, 0.0, 20.0)
        out = self.res_blocks(out)
        return out


class DeepSpeakerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvAndResBlock(1, 64)
        self.block2 = ConvAndResBlock(64, 128)
        self.block3 = ConvAndResBlock(128, 256)
        self.block4 = ConvAndResBlock(256, 512)
        self.affine = nn.Linear(2048, 512)

    def forward(self, x):
        # Input shape: (B, T, F, 1) or (B, T, F)
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        # Permute to PyTorch Conv2D format (B, Channels, Height, Width) -> (B, 1, T, F)
        x = x.permute(0, 3, 1, 2)

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        # Permute back to (B, T_final, F_final, Channels)
        out = out.permute(0, 2, 3, 1)
        B, T_final, F_final, C = out.shape
        out = out.reshape(B, T_final, F_final * C)  # (B, T_final, 2048)

        # Temporal Average over time dimension (dim=1)
        out = torch.mean(out, dim=1)  # (B, 2048)

        out = self.affine(out)

        # L2 Normalize
        out = F.normalize(out, p=2, dim=1)
        return out


def load_weights_from_h5(model, h5_path):
    import h5py
    import numpy as np

    logger.info(f"Loading weights from legacy Keras H5 file: {h5_path} into PyTorch model...")
    with h5py.File(h5_path, 'r') as f:
        def get_data(layer_name, weight_name):
            for prefix in ['', 'model_weights/']:
                path = f"{prefix}{layer_name}/{layer_name}/{weight_name}"
                if path in f:
                    return np.array(f[path])
                path2 = f"{prefix}{layer_name}/{weight_name}"
                if path2 in f:
                    return np.array(f[path2])
            raise KeyError(f"Weight {weight_name} for layer {layer_name} not found in h5 file.")

        blocks = [model.block1, model.block2, model.block3, model.block4]
        filters_list = [64, 128, 256, 512]

        for stage_idx, (block, filters) in enumerate(zip(blocks, filters_list, strict=False), start=1):
            conv_name = f"conv{filters}-s"
            bn_name = f"{conv_name}_bn"

            # Conv weight mapping (transpose TF kernel shape (H, W, in, out) to PT shape (out, in, H, W))
            w_conv = get_data(conv_name, 'kernel:0')
            b_conv = get_data(conv_name, 'bias:0')
            block.conv.weight.data = torch.from_numpy(w_conv.transpose(3, 2, 0, 1)).float()
            block.conv.bias.data = torch.from_numpy(b_conv).float()

            # BN weight mapping
            block.bn.weight.data = torch.from_numpy(get_data(bn_name, 'gamma:0')).float()
            block.bn.bias.data = torch.from_numpy(get_data(bn_name, 'beta:0')).float()
            block.bn.running_mean.data = torch.from_numpy(get_data(bn_name, 'moving_mean:0')).float()
            block.bn.running_var.data = torch.from_numpy(get_data(bn_name, 'moving_variance:0')).float()

            # Identity blocks
            for i, res_block in enumerate(block.res_blocks):
                branch_name = f"res{stage_idx}_{i}_branch"
                conv_2a = f"{branch_name}_2a"
                bn_2a = f"{conv_2a}_bn"
                conv_2b = f"{branch_name}_2b"
                bn_2b = f"{conv_2b}_bn"

                # conv1
                w_c1 = get_data(conv_2a, 'kernel:0')
                b_c1 = get_data(conv_2a, 'bias:0')
                res_block.conv1.weight.data = torch.from_numpy(w_c1.transpose(3, 2, 0, 1)).float()
                res_block.conv1.bias.data = torch.from_numpy(b_c1).float()
                # bn1
                res_block.bn1.weight.data = torch.from_numpy(get_data(bn_2a, 'gamma:0')).float()
                res_block.bn1.bias.data = torch.from_numpy(get_data(bn_2a, 'beta:0')).float()
                res_block.bn1.running_mean.data = torch.from_numpy(get_data(bn_2a, 'moving_mean:0')).float()
                res_block.bn1.running_var.data = torch.from_numpy(get_data(bn_2a, 'moving_variance:0')).float()

                # conv2
                w_c2 = get_data(conv_2b, 'kernel:0')
                b_c2 = get_data(conv_2b, 'bias:0')
                res_block.conv2.weight.data = torch.from_numpy(w_c2.transpose(3, 2, 0, 1)).float()
                res_block.conv2.bias.data = torch.from_numpy(b_c2).float()
                # bn2
                res_block.bn2.weight.data = torch.from_numpy(get_data(bn_2b, 'gamma:0')).float()
                res_block.bn2.bias.data = torch.from_numpy(get_data(bn_2b, 'beta:0')).float()
                res_block.bn2.running_mean.data = torch.from_numpy(get_data(bn_2b, 'moving_mean:0')).float()
                res_block.bn2.running_var.data = torch.from_numpy(get_data(bn_2b, 'moving_variance:0')).float()

        # Affine Layer (transpose Dense shape (in, out) to PT shape (out, in))
        w_affine = get_data('affine', 'kernel:0')
        b_affine = get_data('affine', 'bias:0')
        model.affine.weight.data = torch.from_numpy(w_affine.transpose(1, 0)).float()
        model.affine.bias.data = torch.from_numpy(b_affine).float()
        logger.info("Successfully loaded all legacy Keras weights into PyTorch model.")
