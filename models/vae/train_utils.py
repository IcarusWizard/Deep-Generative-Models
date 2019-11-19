from .vae import VAE
from .convvae import CONV_VAE
import torch
from tqdm import tqdm

from ..utils import step_loader, nats2bits

def config_model(args, model_param):
    if args.model == 'VAE':
        model_param.update({
            "features" : args.features, 
            "hidden_layers" : args.hidden_layers,
            "output_type" : args.output_type,
            "use_mce" : args.use_mce,
        })
        model = VAE(**model_param)        
    elif args.model == 'CONV-VAE':
        model_param.update({
            "features" : args.features, 
            "down_sampling" : args.down_sampling,
            "res_layers" : args.hidden_layers,
            "output_type" : args.output_type,
            "use_mce" : args.use_mce,
        })        
        model = CONV_VAE(**model_param)
    elif args.model == 'VQ-VAE':
        pass
    else:
        raise ValueError('Model {} is not supported!'.format(args.model))

    return model, model_param

def train_vae(model, optim, writer, train_loader, val_loader, test_loader, args):
    device = next(model.parameters()).device
    step = 0
    with tqdm(total=args.steps) as timer:
        for batch in step_loader(train_loader):
            batch = batch[0].to(device)

            loss, kl, reconstruction_loss = model(batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % args.log_step == 0:
                val_loss, val_kl, val_reconstruction_loss = test_vae(model, val_loader)

                print('In Step {}'.format(step))
                print('-' * 15)
                print('In training set:')
                print('NELOB is {0:{1}} bits'.format(nats2bits(loss.item()), '.5f'))
                print('KL divergence is {0:{1}} bits'.format(nats2bits(kl.item()), '.5f'))
                print('Reconstruction loss is {0:{1}} bits'.format(nats2bits(reconstruction_loss.item()), '.5f'))
                print('In validation set:')
                print('NELOB is {0:{1}} bits'.format(nats2bits(val_loss.item()), '.5f'))
                print('KL divergence is {0:{1}} bits'.format(nats2bits(val_kl.item()), '.5f'))
                print('Reconstruction loss is {0:{1}} bits'.format(nats2bits(val_reconstruction_loss.item()), '.5f'))

                writer.add_scalars('NELOB', {'train' : nats2bits(loss.item()), 
                                                'val' : nats2bits(val_loss.item())}, global_step=step)
                writer.add_scalars('KL divergence', {'train' : nats2bits(kl.item()), 
                                                        'val' : nats2bits(val_kl.item())}, global_step=step)
                writer.add_scalars('Reconstruction loss', {'train' : nats2bits(reconstruction_loss.item()), 
                                                            'val' : nats2bits(val_reconstruction_loss.item())}, global_step=step)
                imgs = torch.clamp(model.sample(128, deterministic=True), 0, 1)
                writer.add_images('samples', imgs, global_step=step)

            step = step + 1
            timer.update(1)

            if step >= args.steps:
                break

    test_loss, test_kl, test_reconstruction_loss = test_vae(model, test_loader)
    print('In test set:')
    print('NELOB is {0:{1}} bits'.format(nats2bits(test_loss.item()), '.5f'))
    print('KL divergence is {0:{1}} bits'.format(nats2bits(test_kl.item()), '.5f'))
    print('Reconstruction loss is {0:{1}} bits'.format(nats2bits(test_reconstruction_loss.item()), '.5f'))

    return model, test_loss

def test_vae(model, loader):
    device = next(model.parameters()).device

    with torch.no_grad():
        num = 0
        loss = 0
        kl = 0
        reconstruction_loss = 0
        for batch in iter(loader):
            num += 1
            batch = batch[0].to(device)
            _loss, _kl, _reconstruction_loss = model(batch)
            loss += _loss
            kl += _kl
            reconstruction_loss += _reconstruction_loss 
        loss = loss / num
        kl = kl / num
        reconstruction_loss = reconstruction_loss / num

    return loss, kl, reconstruction_loss