from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def get_dataloader(dataset, args):
    transform_img = transforms.Compose([
                        transforms.Resize(size=self.image_size, interpolation=0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ])
    
    if dataset == 'mnist': 
        dataset = dataset.MNIST(root='./DATA', 
                                trian=True,
                                transform=transform_img,
                                download=True)
    elif dataset == 'fashionmnist':
        dataset = datasets.fashionmnist(root='../DATA', 
                                        train=True, 
                                        download=True,
                                        transform=transform_img)
    elif dataset == 'celebA' : 
        dataset = datasets.CelebA(root='../DATA', 
                                  split='train',
                                  download=True, 
                                  transform=transform_img)

    loader = DataLoader(dataset=dataset, 
                        batch_size=args.batch, 
                        shuffle=True, 
                        num_workers=4)
    return loader 