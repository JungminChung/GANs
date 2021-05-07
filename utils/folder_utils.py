import os 

def exist_or_make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def get_working_dir_name(root_dir, args):
    folders = os.listdir(root_dir)
    num_list = [int(f.split('_')[-1]) for f in folders if f.split('_')[0] == args.model]
    if len(num_list) > 0 : 
        num = max(num_list) + 1 
    else : 
        num = 0
    return f'{args.model}_{str(num)}'

def make_subfolders(working_dir, mode):
    if mode == 'train': 
        exist_or_make_dir(os.path.join(working_dir, 'weights'))
        exist_or_make_dir(os.path.join(working_dir, 'eval_img'))
    else  : 
        exist_or_make_dir(os.path.join(working_dir, 'results_img'))

def make_results_folder(mode, args): 
    mid_folder = f'{mode}_{args.dataset}_results'
    dir_name = get_working_dir_name(tt, args)
    save_folder = os.path.join(mid_folder, dir_name)
    exist_or_make_dir(save_folder)
    make_subfolders(save_folder, mode)

    if mode == 'train' : 
        return save_folder, os.path.join(save_folder, 'weights'), os.path.join(save_folder, 'eval_img')
    else : 
        return save_folder, os.path.join(save_folder, 'results_img')