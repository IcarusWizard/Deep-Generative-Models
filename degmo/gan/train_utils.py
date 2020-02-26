from .vanilla_gan import GAN
from .dcgan import DCGAN
from .wgan import WGAN
from .wgangp import WGANGP
from .sngan import SNGAN
from .sagan import SAGAN

import torch
from tqdm import tqdm

from ..utils import step_loader

def config_model(args, model_param):
    if args.model == 'GAN':
        model_param.update({
            "latent_dim" : args.latent_dim,
            "discriminator_features" : args.discriminator_features, 
            "discriminator_hidden_layers" : args.discriminator_hidden_layers,
            "generator_features" : args.generator_features,
            "generator_hidden_layers" : args.generator_hidden_layers,
        })
        model = GAN(**model_param)  
    elif args.model == 'DCGAN':
        model_param.update({
            "latent_dim" : args.latent_dim,
            "discriminator_features" : args.discriminator_features, 
            "generator_features" : args.generator_features,
        })              
        model = DCGAN(**model_param)
    elif args.model == 'WGAN':
        model_param.update({
            "mode" : args.mode,
            "latent_dim" : args.latent_dim,
            "discriminator_features" : args.discriminator_features, 
            "discriminator_hidden_layers" : args.discriminator_hidden_layers,
            "generator_features" : args.generator_features,
            "generator_hidden_layers" : args.generator_hidden_layers,
            "use_norm_discriminator" : args.use_norm_discriminator,
            "use_norm_generator" : args.use_norm_generator,
        })
        model = WGAN(**model_param) 
    elif args.model == 'WGAN-GP':
        model_param.update({
            "mode" : args.mode,
            "latent_dim" : args.latent_dim,
            "_lambda" : args._lambda,
            "discriminator_features" : args.discriminator_features, 
            "discriminator_hidden_layers" : args.discriminator_hidden_layers,
            "generator_features" : args.generator_features,
        })
        model = WGANGP(**model_param)      
    elif args.model == 'SNGAN':
        model_param.update({
            "mode" : args.mode,
            "latent_dim" : args.latent_dim,
            "discriminator_features" : args.discriminator_features, 
            "discriminator_hidden_layers" : args.discriminator_hidden_layers,
            "generator_features" : args.generator_features,
        })
        model = SNGAN(**model_param)         
    elif args.model == 'SAGAN':
        model_param.update({
            "latent_dim" : args.latent_dim,
            "discriminator_features" : args.discriminator_features, 
            "discriminator_hidden_layers" : args.discriminator_hidden_layers,
            "generator_features" : args.generator_features,
        })
        model = SAGAN(**model_param)       
    else:
        raise ValueError('Model {} is not supported!'.format(args.model))

    return model, model_param

def train_gan(model, writer, train_loader, args):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # config optimizor
    discriminator_optim = torch.optim.Adam(model.discriminator.parameters(), 
        lr=args.discriminator_lr, betas=(args.discriminator_beta1, args.discriminator_beta2))
    generator_optim = torch.optim.Adam(model.generator.parameters(), 
        lr=args.generator_lr, betas=(args.generator_beta1, args.generator_beta2))

    z = torch.randn(64, args.latent_dim, device=device, dtype=dtype)
    
    step = 0
    loader = step_loader(train_loader)
    with tqdm(total=args.steps) as timer:
        while step < args.steps:
            # train discriminator for n_critic times
            for i in range(args.n_critic):
                real = next(loader)[0].to(device)
                fake = model.generate(real.shape[0])

                discriminator_loss = model.get_discriminator_loss(real, fake)

                discriminator_optim.zero_grad()
                discriminator_loss.backward()
                discriminator_optim.step()

                if args.model == 'WGAN':
                    with torch.no_grad():
                        for name, value in model.discriminator.named_parameters():
                            if 'weight' in name:
                                value.clamp_(-args.clamp, args.clamp)

            # train generator only once
            fake = model.generate(real.shape[0])
            generator_loss = model.get_generator_loss(fake)

            generator_optim.zero_grad()
            generator_loss.backward()
            generator_optim.step()

            if step % args.log_step == 0:
                print('In Step {}'.format(step))
                print('-' * 15)
                print('Discriminator loss is {0:{1}}'.format(discriminator_loss.item(), '.3f'))
                print('Generator loss is {0:{1}}'.format(generator_loss.item(), '.3f'))

                writer.add_scalars('loss', {'discriminator' :discriminator_loss.item(), 
                                            'generator' : generator_loss.item()}, global_step=step)

                imgs = torch.clamp(model.z2x(z) / 2 + 0.5, 0, 1)
                writer.add_images('fixed_samples', imgs, global_step=step)

                imgs = torch.clamp(model.generate(64) / 2 + 0.5, 0, 1)
                writer.add_images('samples', imgs, global_step=step)

            step = step + 1
            timer.update(1)

    return model, discriminator_optim, generator_optim