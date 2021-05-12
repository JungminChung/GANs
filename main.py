import os
import sys
import tqdm
import argparse
import torch 

from utils.check_args import check_args
from utils.get_dataloader import get_dataloader
from utils.get_tools import get_model, get_losses
from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='vanillagan', help='vanillagan / dcgan / cgan / wgan / wgangp')
    parser.add_argument('--dataset', type=str, default='mnist', help='mnist / fashionmnist / celeA')
    parser.add_argument('--mode', type=str, default='train', help='train / test')
    
    parser.add_argument('--device', type=str)
    
    # train 
    parser.add_argument('--save_subfolder_path', type=str, default='')
    parser.add_argument('--save_weight_path', type=str, default='')
    parser.add_argument('--save_train_img_path', type=str, default='')
    parser.add_argument('--batch', type=int, default=64, help='mini batch size')
    parser.add_argument('--step', type=int, default=100_000, help='total step for training')
    parser.add_argument('--save_step', type=int, default=5_000, help='interval to save steps')

    parser.add_argument('--no_recommend_setting', action='store_false') 
    parser.add_argument('--z_dim', type=int, default=100, help='latent random number size')
    parser.add_argument('--ngf', type=int, default=256, help='hidden layer size in Generator')
    parser.add_argument('--gen_img_ch', type=int, default=1, help='generated image channel size')
    parser.add_argument('--ndf', type=int, default=256, help='hidden layer size in Discriminator')
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--n_critic', type=int, default=1, help='ratio of number of learners to generator and discriminator ')

    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='adam beta 1')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam beta 2')
    
    # test 
    parser.add_argument('--save_test_img_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--label', type=int, default=1, help='label value for test. Only valid on conditional setting')


    return parser.parse_args()

def main():
    sys.path.append('.')
    
    args = parse_args()
    args = check_args(args)
    print(args)

    if args.mode == 'train' : 
        G, D = get_model(args.model, args)
        G_loss, D_loss = get_losses(args.model, args)
        G_optim = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        D_optim = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        loader = iter(get_dataloader(args.dataset, args))
        
        summary = SummaryWriter(args.save_subfolder_path)
        pbar = tqdm(range(args.step))
        epoch = 0

        fixed_l = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9], dtype=torch.float32).to(device)
        fixed_z = torch.randn(fixed_l.shape[0], args.z_dim, 1, 1).to(args.device)

        for step in pbar :
            try : 
                real_img, label = next(loader)

            except : 
                loader = iter(loader)
                real_img, label = next(loader)
                    
                epoch += 1 

            real_img = real_img.to(args.device)
            label = label.to(args.device)

            d_loss, d_real_loss, d_fake_loss = update_D(D, G, real_img, label, D_optim, D_loss, args)
            g_loss = update_G(D, G, label, G_optim, G_loss, args)

            if step % args.save_step == 0: 
                # save checkpoints 
                save_ckpt(args.save_weight_path, G, D, step, epoch) 

                # save middle point images 
                fake_img = G(fixed_z)
                save_img(args.save_train_img_path, fake_img, step) 

                # save summaries 
                write_logs(summary, d_loss, d_real_loss, d_fake_loss, g_loss, step)

            state_msg = (
                f'{epoch} Epo / '+ 
                f'D_real : {d_real_loss:0.3f} ; D_fake : {d_fake_loss:0.3f} ; ' + 
                f'D : {d_loss:0.3f} ; G : {g_loss:0.3f} ; ' 
            )
            pbar.set_description(state_msg)

    else : 
        G = get_model(args.model, args)
        pass

    
if __name__=='__main__':
    main()
