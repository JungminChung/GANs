from .folder_utils import make_results_folder

def check_args(args):
    if args.dataset == 'celebA' and args.model == 'cgan' : 
        raise AssertionError('didn\'t provide celebA dataset on cgan')
    
    if args.mode == 'train' : 
        args.save_subfolder_path, args.save_weight_path, args.save_train_img_path = make_results_folder(args.mode, args)
    else : 
        args.save_subfolder_path, args.save_test_img_path = make_results_folder(args.mode, args)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    