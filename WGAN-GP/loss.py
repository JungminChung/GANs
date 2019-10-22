import torch
import torch.nn as nn 

class GAN_D_loss(nn.Module):
    def __init__(self):
        super(GAN_D_loss, self).__init__()

    def forward(self, D_real, D_fake):
        d_loss = -torch.mean(D_real)+torch.mean(D_fake)

        return d_loss

class GAN_G_loss(nn.Module):
    def __init__(self):
        super(GAN_G_loss, self).__init__()

    def forward(self, D_fake):
        g_loss = -torch.mean(D_fake)

        return g_loss