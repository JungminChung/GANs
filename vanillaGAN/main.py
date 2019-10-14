from vanillaGAN import vanillaGAN
import argparse
import utils
import torch 

def parse_args():
    parser = argparse.ArgumentParser()

    # Basic setting 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device)
    print('Running Device :', device)
    parser.add_argument('--mode', type=str, default='train', help='train / test')

    # Folder setting 
    ckpt_folder, image_folder, source_folder, summary_folder = utils.folder_setting()
    parser.add_argument('--ckpt_dir', type=str, default=ckpt_folder)
    parser.add_argument('--img_dir', type=str, default=image_folder)
    parser.add_argument('--scr_dir', type=str, default=source_folder)
    parser.add_argument('--sum_dir', type=str, default=summary_folder, help='tensorboard log write iteration')
    
    # Dataset setting 
    parser.add_argument('--mnist', type=bool, default=True, help='train for mnist dataset?')
    parser.add_argument('--mnist_path', type=str, default='../DATASETS/mnist')
    
    # Network setting 
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='worker number for data load')
    parser.add_argument('--total_step', type=int, default=100_000, help='total step for training')
    parser.add_argument('--image_size', type=int, default=28, help='input image size, 28 X 28 ')
    
    # Generator 
    parser.add_argument('--z_dim', type=int, default=100, help='latent random number size')
    parser.add_argument('--ngf', type=int, default=256, help='hidden layer size in Generator')

    # Discriminator 
    parser.add_argument('--ndf', type=int, default=256, help='hidden layer size in Discriminator')

    # Adam Optimizer 
    parser.add_argument('--lr_G', type=float, default=0.0002, help='initial adam learning rate for Generator')
    parser.add_argument('--lr_D', type=float, default=0.0002, help='initial adam learning rate for Discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='initial adam beta 1')
    parser.add_argument('--beta2', type=float, default=0.999, help='initial adam beta 2')
    
    # Save setting
    parser.add_argument('--n_save', type=int, default=200, help='save interval btw save check points on step')
    parser.add_argument('--n_save_image', type=int, default=200, help='save interval btw save images on step')
    parser.add_argument('--n_summary', type=int, default=10, help='save interval btw save summary points on step')
    
    # Etc 
    parser.add_argument('--n_show_img', type=int, default=128, help='how many images you want to save at once')

    # Test 
    parser.add_argument('--trained_dir', type=str, default='./pre_trained')
    parser.add_argument('--trained_model_name', type=str, default='mnist')
    parser.add_argument('--test_dir', type=str, default='./test_output')

    return parser.parse_args()

def main():
    args = parse_args()
    if not args :
        exit()

    GAN = vanillaGAN(args)
    
    print('build model')
    GAN.build_model()
    
    if args.mode == 'train' : 
        print('load datasets')
        GAN.load_dataset()

        print('train start')
        GAN.train()

    elif args.mode == 'test': 
        print('test mode')
        GAN.test()

    else : 
        print('mode will be train or test')
    
if __name__=='__main__':
    main()
