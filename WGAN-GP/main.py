from WGANGP import WGANGP
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
    parser.add_argument('--mnist', type=bool, default=False, help='train for mnist dataset?')
    parser.add_argument('--mnist_path', type=str, default='../DATASETS/mnist')
    parser.add_argument('--celeb_path', type=str, default='../DATASETS/celeb')
    
    # Network setting 
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='worker number for data load')
    parser.add_argument('--total_step', type=int, default=1_000_000, help='total step for training')
    
    # Generator 
    parser.add_argument('--image_size', type=int, default=64, help='generated image size')
    parser.add_argument('--channel', type=int, default=3, help='generated image channel size')
    parser.add_argument('--z_dim', type=int, default=100, help='latent random number size')
    parser.add_argument('--init_height', type=int, default=4, help='initial height of feature map in Generator')
    parser.add_argument('--init_width', type=int, default=4, help='initial width of feature map in Generator')
    parser.add_argument('--ngf', type=int, default=128, help='basic feature channel size in Generator')

    # Discriminator 
    parser.add_argument('--n_critic', type=int, default=5, help='ratio of number of learners to generator and discriminator ')
    parser.add_argument('--ndf', type=int, default=32, help='basic feature channel size in Discriminator')
    parser.add_argument('--lambda_GP', type=int, default=10, help='weight of gp')

    # Adam Optimizer 
    parser.add_argument('--lr_G', type=float, default=0.0001, help='initial adam learning rate for Generator')
    parser.add_argument('--lr_D', type=float, default=0.0001, help='initial adam learning rate for Discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='initial beta 1 of adam learning rate for Discriminator')
    parser.add_argument('--beta2', type=float, default=0.9, help='initial beta 2 adam learning rate for Discriminator')
    
    # Save setting
    parser.add_argument('--n_save', type=int, default=5_000, help='save interval btw save check points on step')
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

    wgangp = WGANGP(args)
    
    print('build model')
    wgangp.build_model()
    
    if args.mode == 'train' : 
        print('load datasets')
        wgangp.load_dataset()

        print('train start')
        wgangp.train()

    elif args.mode == 'test': 
        print('test mode')
        wgangp.test()

    else : 
        print('mode will be train or test')
    
if __name__=='__main__':
    main()
