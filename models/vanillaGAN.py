import torch 
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, ngf, image_size):
        super(Generator, self).__init__()
        self.z_dim = z_dim 
        self.ngf = ngf
        self.image_size = image_size

        self.G = nn.Sequential(
            nn.Linear(in_features=self.z_dim, out_features=self.ngf, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.ngf, out_features=self.ngf, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.ngf, out_features=self.image_size * self.image_size, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, x.size()[1])
        x = self.G(x)
        x = x.view(-1, 1, self.image_size, self.image_size)

        return x

class Discriminator(nn.Module):
    def __init__(self, ndf, image_size):
        super(Discriminator, self).__init__()

        self.image_size = image_size 
        self.ndf = ndf

        self.D = nn.Sequential(
            nn.Linear(in_features=self.image_size*self.image_size, out_features=self.ndf, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=self.ndf, out_features=self.ndf, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=self.ndf, out_features=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, self.image_size * self.image_size)
        x = self.D(x)

        return x
