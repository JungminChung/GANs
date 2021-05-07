import torch 
from torchvision.utils import save_image

def update_D(D, G, real_img, label, D_optim, D_loss, args):
    D_optim.zero_grad()

    z = torch.randn(args.batch, args.z_dim, 1, 1).to(args.device)
    fake_img = G(z)

    D_real = D(real_img)
    D_fake = D(fake_img)
    
    d_loss, d_real_loss, d_fake_loss = D_loss(D_real, D_fake)

    d_loss.backward()
    D_optim.step()
    
    return d_loss, d_real_loss, d_fake_loss

def update_G(D, G, label, G_optim, G_loss, args):
    G_optim.zero_grad() 

    z = torch.randn(args.batch, args.z_dim, 1, 1).to(args.device)
    fake_img = G(z)
    
    D_fake = D(fake_img)

    g_loss = G_loss(D_fake)
    
    g_loss.backward()
    G_optim.step()
    
    return g_loss

def save_ckpt(dir, G, D, step, epoch):
    model_dict = {} 
    model_dict['gen'] = G.state_dict()
    model_dict['dis'] = D.state_dict()
    torch.save(model_dict, 
               os.path.join(dir, f'{str(step+1).zfill(7)}.ckpt'))


def save_img(dir, img, step):
    with torch.no_grad():
        save_image(denorm(img), 
                   os.path.join(dir, f'{str(step+1).zfill(7)}.jpg'))

def denorm(x):
    out = (x+1)/2
    return out.clamp(0, 1)

def write_logs(summary, d_loss, d_real_loss, d_fake_loss, g_loss, step):
    summary.add_scalar('D', d_loss.item(), step)
    summary.add_scalar('D_real', d_real_loss.item(), step)
    summary.add_scalar('D_fake', d_fake_loss.item(), step)
    summary.add_scalar('G', g_loss.item(), step)
    