import torch 
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, ngf, gen_img_ch, n_label=10):
        super(Generator, self).__init__()
        self.z_dim = z_dim 
        self.n_label = n_label
        self.gen_img_ch = gen_img_ch
        self.ngf = ngf

        self.projectionZ = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=self.ngf*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf*4), 
            nn.ReLU(inplace=True),
        )

        self.projectionL = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_label, out_channels=self.ngf*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf*4),  
            nn.ReLU(inplace=True),
        )

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels= self.ngf*8, out_channels=self.ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(inplace=True),
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ngf*4, out_channels=self.ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(inplace=True),
        )
        
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ngf*2, out_channels=self.ngf*1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf*1),
            nn.ReLU(inplace=True),
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ngf*1, out_channels=self.gen_img_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        
    def forward(self, z, label):
        z = self.projectionZ(z)
        l = self.projectionL(label)

        x = torch.cat((z,l), dim=1)

        x = self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        x = self.convT4(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, ndf, image_size, image_channel, n_label=10):
        super(Discriminator, self).__init__()

        self.ndf = ndf
        self.image_size = image_size
        self.image_channel = image_channel 
        self.n_label = n_label

        self.conv1X = nn.Sequential(
            nn.Conv2d(in_channels=self.image_channel, out_channels=self.ndf*1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv1L = nn.Sequential(
            nn.Conv2d(in_channels=self.n_label, out_channels=self.ndf*1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ndf*2, out_channels=self.ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ndf*4),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ndf*4, out_channels=self.ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ndf*8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ndf*8, out_channels=self.ndf*16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.ndf*16),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=self.ndf*16, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False), 
            nn.Sigmoid(),
        )

    def forward(self, x, label):
        if x.size()[1] == 1: 
            x = torch.cat((x,x,x), dim=1)

        l = label.expand(label.size(0), self.n_label, self.image_size, self.image_size)
        x = self.conv1X(x)
        l = self.conv1L(l)

        x = torch.cat((x,l), dim=1)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) 
        x = self.out(x) 

        return x

def _init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear): 
        nn.init.normal_(tensor=m.weight.data, mean=0, std=0.02)
        if m.bias is not None : 
            nn.init.constant_(tensor=m.bias.data, val=0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(tensor=m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(tensor=m.bias.data, val=0)