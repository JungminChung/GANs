import torch 
import torch.nn as nn 

class vanilla_G_loss(nn.Module):
    def __init__(self, device):
        super(vanilla_G_loss, self).__init__()
        self.device = device 
        self.criterion = nn.BCELoss()

    def forward(self, fake):
        ones = torch.ones_like(fake).to(self.device)

        g_loss = self.criterion(fake, ones)

        return g_loss

class vanilla_D_loss(nn.Module):
    def __init__(self, device):
        super(vanilla_D_loss, self).__init__()
        self.device = device
        self.criterion = nn.BCELoss()

    def forward(self, D_real, D_fake):
        ones = torch.ones_like(D_real).to(self.device)
        zeros = torch.zeros_like(D_fake).to(self.device)

        d_real_loss = self.criterion(D_real, ones)
        d_fake_loss = self.criterion(D_fake, zeros)

        d_loss = d_real_loss + d_fake_loss

        return d_loss, d_real_loss, d_fake_loss


class wasserstein_G_loss(nn.Module):
    def __init__(self):
        super(wasserstein_G_loss, self).__init__()

    def forward(self, D_fake):
        g_loss = -torch.mean(D_fake)

        return g_loss

class wasserstein_D_loss(nn.Module):
    def __init__(self):
        super(wasserstein_D_loss, self).__init__()

    def forward(self, D_real, D_fake):
        d_loss = -(torch.mean(D_real)-torch.mean(D_fake))

        return d_loss

