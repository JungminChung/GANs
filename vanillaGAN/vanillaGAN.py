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

class vanillaGAN(object):
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
        self.image_size = args.image_size

        self.z_dim = args.z_dim
        self.ngf = args.ngf
        
        self.ndf = args.ndf
        
        self.lr_G = args.lr_G
        self.lr_D = args.lr_D
        self.beta1 = args.beta1
        self.beta2 = args.beta2

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
        self.transform_img = transforms.Compose([
                                transforms.Resize(size=self.image_size, interpolation=0),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                            ])

        self.dataset = datasets.MNIST(root=self.mnist_path,
                                        train=True,
                                        transform=self.transform_img,
                                        download=True
                                    )

        self.loader = DataLoader(dataset=self.dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    num_workers=self.num_workers,
                                )

    def build_model(self): 
        ########## Networks ##########
        self.gen = DP(Generator(self.z_dim, self.ngf, self.image_size)).to(self.device)
        self.dis = DP(Discriminator(self.ndf, self.image_size)).to(self.device)

        ########## Loss ##########
        self.GAN_D_loss = GAN_D_loss(self.device)
        self.GAN_G_loss = GAN_G_loss(self.device)

        ########## Optimizer ##########
        self.G_optim = torch.optim.Adam(self.gen.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))
        self.D_optim = torch.optim.Adam(self.dis.parameters(), lr=self.lr_D, betas=(self.beta1, self.beta2))

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
                real_image, _ = next(data_loader)

            except : 
                data_loader = iter(self.loader)
                real_image, _ = next(data_loader)
                    
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
                'D_real : {:0.3f} ; D_fake : {:0.3f} ; '.format(self.d_real_loss, self.d_fake_loss) + 
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

        self.d_loss, self.d_real_loss, self.d_fake_loss = self.GAN_D_loss(self.D_real, self.DD_fake)
        
        self.D_real = self.D_real.mean().item()
        self.DD_fake = self.DD_fake.mean().item()

        self.d_loss.backward()
        self.D_optim.step()
    
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
        summary.add_scalar('D_real', self.d_real_loss.item(), step)
        summary.add_scalar('D_fake', self.d_fake_loss.item(), step)
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
