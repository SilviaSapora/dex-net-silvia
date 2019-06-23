import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


class GGCNN4(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None, **kwargs):
        super(GGCNN4, self).__init__(**kwargs)

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
        
        pos_pred_for_loss = pos_pred.clone()
        cos_pred_for_loss = cos_pred.clone()
        sin_pred_for_loss = sin_pred.clone()
        y_pos_for_loss = y_pos.clone()
        ignore_idx = (torch.where(y_pos == -1, torch.tensor([True]),
                torch.tensor([False])))
        pos_pred_for_loss[ignore_idx] = 0
        y_pos_for_loss[ignore_idx] = 0
        cos_pred_for_loss[ignore_idx] = -10
        sin_pred_for_loss[ignore_idx] = -10
        
        pos_pred_for_loss_n = pos_pred_for_loss.detach().numpy().reshape(200,200)
        cos_pred_for_loss_n = cos_pred_for_loss.detach().numpy().reshape(200,200)
        sin_pred_for_loss_n = sin_pred_for_loss.detach().numpy().reshape(200,200)
        y_pos_n = y_pos_for_loss.detach().numpy().reshape(200,200)
        y_cos_n = y_cos.detach().numpy().reshape(200,200)
        y_sin_n = y_sin.detach().numpy().reshape(200,200)

        plt.figure()
        plt.subplot(231)
        plt.imshow(pos_pred_for_loss_n[75:125, 75:125])
        plt.colorbar()
        plt.subplot(232)
        plt.imshow(cos_pred_for_loss_n[75:125, 75:125])
        plt.colorbar()
        plt.subplot(233)
        plt.imshow(sin_pred_for_loss_n[75:125, 75:125])
        plt.colorbar()
        plt.subplot(234)
        plt.imshow(y_pos_n[75:125, 75:125])
        plt.colorbar()
        plt.subplot(235)
        plt.imshow(y_cos_n[75:125, 75:125])
        plt.colorbar()
        plt.subplot(236)
        plt.imshow(y_sin_n[75:125, 75:125])
        plt.colorbar()
        plt.show()

        p_loss = F.binary_cross_entropy(pos_pred_for_loss, y_pos_for_loss)
        #p_loss = F.binary_cross_entropy(pos_pred, y_pos_for_loss)
        cos_loss = F.mse_loss(cos_pred_for_loss, y_cos)
        #cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred_for_loss, y_sin)
        #sin_loss = F.mse_loss(sin_pred, y_sin)

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
