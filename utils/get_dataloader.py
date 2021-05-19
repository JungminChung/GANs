from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def get_dataloader(dataset, args):
    if dataset == 'celebA' : 
        m = (0.5, 0.5, 0.5)
        s = (0.5, 0.5, 0.5)
    else : 
        m = (0.5, )
        s = (0.5, )
    transform_img = transforms.Compose([
                        transforms.Resize(size=args.img_size, interpolation=0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = m, std = s),
                    ])
    
    if dataset == 'mnist': 
        dataset = datasets.MNIST(root='./DATA', 
                                train=True,
                                transform=transform_img,
                                download=True)
    elif dataset == 'fashionmnist':
        dataset = datasets.FashionMNIST(root='./DATA', 
                                        train=True, 
                                        download=True,
                                        transform=transform_img)
    elif dataset == 'celebA' : 
        dataset = datasets.CelebA(root='./DATA', 
                                  split='train',
                                  download=True, 
                                  transform=transform_img)

    loader = DataLoader(dataset=dataset, 
                        batch_size=args.batch, 
                        shuffle=True, 
                        num_workers=4)
    return loader 