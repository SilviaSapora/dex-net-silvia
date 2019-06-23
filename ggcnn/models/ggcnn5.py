import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GGCNN5(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None, **kwargs):
        super(GGCNN5, self).__init__(**kwargs)

        if filter_sizes is None:
            filter_sizes = [16,  # First set of convs
                            16,  # Second set of convs
                            32,  # Dilated convs
                            16]  # Transpose Convs

        if dilations is None:
            dilations = [2, 4]

        self.features = nn.Sequential(
            # 4 conv layers.
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions.
            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
            nn.ReLU(inplace=True),

            # Output layers
            nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(filter_sizes[3], filter_sizes[3], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),

        )

        self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.features(x)

        pos_output = torch.sigmoid(self.pos_output(x))
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)

        return pos_output, cos_output, sin_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin = yc
        pos_pred, cos_pred, sin_pred = self(xc)

        # y_pos = y_pos.reshape((8, 200,200))
        # ignore_idx = torch.where(y_pos == -1, 1, 0)
        # pos_pred[ignore_idx] = -1
        
        pos_pred_for_loss = pos_pred.clone()
        ignore_idx = (torch.where(y_pos == -1, torch.tensor([True]).cuda(), torch.tensor([False]).cuda())).cuda()
        # ignore_idx = ignore_idx.reshape((8,1,200,200))
        pos_pred_for_loss[ignore_idx] = -1

        # mask = torch.zeros(6, 1, 25)
        # mask.scatter_(2, indices, 1.)

        p_loss = F.binary_cross_entropy(pos_pred_for_loss, y_pos) * 1000
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)

        return {
            'loss': p_loss + cos_loss + sin_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
            }
        }
