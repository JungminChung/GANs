import os 

from models import vanillaGAN, DCGAN, CGAN, WGAN, WGANGP
from .loss import vanilla_G_loss, vanilla_D_loss, wasserstein_G_loss, wasserstein_D_loss

def get_model(model, args):
    if args.mode == 'train' : 
        if args.no_recommend_setting : 
            print('Use arg parsing settings for training ')
        else : 
            print(f'Change recommendation train settings based on {model} model')
            args = change_train_settings(model, args)
    
    if model == 'vanillagan' : 
        gen = vanillaGAN.Generator(args.z_dim, args.ngf, args.img_size)
        dis = vanillaGAN.Discriminator(args.ndf, args.img_size)
    elif model == 'dcgan' : 
        gen = DCGAN.Generator(args.z_dim, args.ngf, args.gen_img_ch)
        dis = DCGAN.Discriminator(args.ndf, args.gen_img_ch)
    elif model == 'cgan' : 
        gen = CGAN.Generator(args.z_dim, args.ngf, args.gen_img_ch)
        dis = CGAN.Discriminator(args.ndf, args.img_size, args.gen_img_ch)
    elif model == 'wgan' : 
        gen = WGAN.Generator(args.z_dim, args.ngf, args.gen_img_ch)
        dis = WGAN.Discriminator(args.ndf, args.gen_img_ch)
    elif model == 'wgangp' : 
        gen = WGANGP.Generator(args.z_dim, args.ngf, args.gen_img_ch)
        dis = WGANGP.Discriminator(args.ndf, args.gen_img_ch)
    
    if args.mode == 'train' : 
        return gen.to(args.device), dis.to(args.device)
    else : 
        return gen.to(args.device) 

def change_train_settings(model, args):
    if   model == 'vanillagan' : 
        args.z_dim = 100
        args.ngf = 256
        args.ndf = 256

    elif model == 'cgan' : 
        args.z_dim = 100
        args.ngf = 64
        args.gen_img_ch = 3
        args.ndf = 32
        args.img_size = 64

    elif model == 'dcgan' : 
        args.z_dim = 100
        args.ngf = 64
        args.gen_img_ch = 3
        args.ndf = 32
        args.lr = 0.00002

    elif model == 'wgan' : 
        args.z_dim = 100
        args.ngf = 64
        args.gen_img_ch = 3
        args.ndf = 64 
        args.lr = 0.00005
        args.beta2 = 0.9

    elif model == 'wgangp' : 
        args.z_dim = 100
        args.ngf = 64
        args.gen_img_ch = 3
        args.ndf = 64 
        args.lambda_GP = 10
        args.lr = 0.0001
        args.beta2 = 0.9
        args.n_critic = 5 

    return args 

def get_losses(model_name, args): 
    if 'wgan' in model_name : 
        g_loss = wasserstein_G_loss()
        d_loss = wasserstein_D_loss()
    else : 
        g_loss = vanilla_G_loss(args.device)
        d_loss = vanilla_D_loss(args.device)
    
    return g_loss, d_loss