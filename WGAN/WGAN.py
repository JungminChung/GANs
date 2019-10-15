import torch
import os
import utils
import random
import torchvision

import networks
from loss import *
from networks import *
from dataset import *

from tqdm import tqdm 

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from torch.nn import DataParallel as DP

from tensorboardX import SummaryWriter

class WGAN(object):
    def __init__(self, args):
        self.device = args.device
        self.mode = args.mode

        self.ckpt_dir = args.ckpt_dir
        self.img_dir = args.img_dir
        self.scr_dir = args.scr_dir
        self.sum_dir = args.sum_dir
        
        self.mnist = args.mnist
        self.mnist_path = args.mnist_path

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.total_step = args.total_step

        self.z_dim = args.z_dim
        self.init_height = args.init_height
        self.init_width = args.init_width
        self.ngf = args.ngf
        
        self.ndf = args.ndf
        self.clip = args.clip
        
        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.n_save = args.n_save
        self.n_save_image = args.n_save_image
        self.n_summary = args.n_summary

        self.n_show_img = args.n_show_img

        self.trained_dir = args.trained_dir
        self.trained_model_name = args.trained_model_name
        self.test_dir = args.test_dir

        manualSeed = 999
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    def load_dataset(self):
        if self.mnist : 
            self.transform_img = transforms.Compose([
                                    transforms.Resize(size=64, interpolation=0),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                ])
            self.dataset = datasets.MNIST(root=self.mnist_path,
                                            train=True,
                                            transform=self.transform_img,
                                            download=True
                                        )
        else : 
            self.transform_img = transforms.Compose([
                                    transforms.Resize(size=64, interpolation=1),
                                    transforms.CenterCrop(size=64),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                ])
            self.dataset = celebDataset(img_path=self.celeb_path, 
                                        transform=self.transform_img,
                                    )

        self.loader = DataLoader(dataset=self.dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    num_workers=self.num_workers,
                                )

    def build_model(self): 
        ########## Networks ##########
        self.gen = DP(Generator(self.z_dim, self.init_height, self.init_width, self.ngf)).to(self.device)
        self.dis = DP(Discriminator(self.ndf)).to(self.device)
        
        ########## Init Networks ##########
        self.gen.apply(networks._init_weights)
        self.dis.apply(networks._init_weights)

        ########## Loss ##########
        self.GAN_D_loss = GAN_D_loss()
        self.GAN_G_loss = GAN_G_loss()

        ########## Optimizer ##########
        self.G_optim = torch.optim.RMSprop(self.gen.parameters(), lr=self.lr_G)
        self.D_optim = torch.optim.RMSprop(self.dis.parameters(), lr=self.lr_D)

    def train(self):
        self.gen.train()
        self.dis.train()

        summary = SummaryWriter(self.sum_dir)

        data_loader = iter(self.loader)

        pbar = tqdm(range(self.total_step))
        epoch = 0 

        self.fixed_z = torch.randn(self.n_show_img, self.z_dim, 1, 1).to(self.device)

        for step in pbar:
            try : 
                if self.mnist :
                    real_image, _ = next(data_loader)
                else :
                    real_image = next(data_loader)

            except : 
                data_loader = iter(self.loader)
                if self.mnist :
                    real_image, _ = next(data_loader)
                else :
                    real_image = next(data_loader)
                epoch += 1 

            real_image = real_image.to(self.device)

            ##### train Discriminator #####
            self.update_D(real_image)
            
            ##### train Generator #####
            self.update_G()

            ##### save checkpoints #####
            if step % self.n_save == 0 : 
                self.save_ckpt(self.ckpt_dir, step, epoch)
 
            ##### save middle point images #####
            if step % self.n_save_image == 0 : 
                fake_image = self.eval()
                self.save_img(self.img_dir, fake_image, step)
                print()

            ##### save tensorboard summary #####
            if step % self.n_summary == 0: 
                self.writeLogs(summary, step)

            state_msg = (
                '{} Epo/ '.format(epoch) + 
                'D : {:0.3f} ; G : {:0.3f} ; '.format(self.d_loss, self.g_loss) + 
                'D(x) : {:0.3f} ; D(G(z)) : {:0.3f}, {:0.3f} ; '.format(self.D_real, self.DD_fake, self.DG_fake)
            )

            pbar.set_description(state_msg)
            
    def update_D(self, real_image): 
        self.D_optim.zero_grad()

        z = torch.randn(self.batch_size, self.z_dim, 1, 1).to(self.device)
        fake_image = self.gen(z)

        self.D_real = self.dis(real_image)
        self.DD_fake = self.dis(fake_image)

        self.d_loss = self.GAN_D_loss(self.D_real, self.DD_fake)
        
        self.D_real = self.D_real.mean().item()
        self.DD_fake = self.DD_fake.mean().item()

        self.d_loss.backward()
        self.D_optim.step()

        for p in self.dis.parameters():
            p.data.clamp_(-self.clip, self.clip)
    
    def update_G(self):
        self.G_optim.zero_grad()

        z = torch.randn(self.batch_size, self.z_dim,1, 1).to(self.device)
        fake_image = self.gen(z)

        self.DG_fake = self.dis(fake_image)

        self.g_loss = self.GAN_G_loss(self.DG_fake)

        self.DG_fake = self.DG_fake.mean().item()
        self.g_loss.backward()
        self.G_optim.step()

    def writeLogs(self, summary, step):
        summary.add_scalar('D', self.d_loss.item(), step)
        summary.add_scalar('G', self.g_loss.item(), step)
        summary.add_scalar('D(x)', self.D_real, step)
        summary.add_scalar('D(G(z_d))', self.DD_fake, step)
        summary.add_scalar('D(G(z_g))', self.DG_fake, step)

    def save_ckpt(self, dir, step, epoch):
        model_dict = {} 
        model_dict['gen'] = self.gen.state_dict()
        model_dict['dis'] = self.dis.state_dict()
        torch.save(model_dict, os.path.join(dir, f'{str(step+1).zfill(7)}.ckpt'))


    def save_img(self, dir, fake_img, name):
        with torch.no_grad():
            save_image(self.denorm(fake_img), os.path.join(dir, f'{str(name+1).zfill(7)}.jpg'))

    def load(self, dir, name):
        if self.mode == 'train':
            model_dict = torch.load(os.path.join(dir, f'{str(ckpt_step).zfill(7)}.ckpt'))
        else : 
            model_dict = torch.load(os.path.join(dir, str(name)+'.ckpt'))

        self.gen.load_state_dict(model_dict['gen'])
        self.dis.load_state_dict(model_dict['dis'])

    def eval(self):
        fake_image = self.gen(self.fixed_z)

        return fake_image

    def denorm(self, x):
        out = (x+1)/2
        return out.clamp(0, 1)

    def test(self):
        self.load(self.trained_dir, self.trained_model_name)
        z = torch.randn(self.n_show_img, self.z_dim, 1, 1)
        fake_image = self.gen(z)
        save_file_num = len(os.listdir(self.test_dir))-1 # folder has gitignore file by default
        self.save_img(self.test_dir, fake_image, save_file_num)

        print('Image', f'{str(save_file_num+1).zfill(7)}.jpg', 'is saved! Chcek "test_output" folder.')
