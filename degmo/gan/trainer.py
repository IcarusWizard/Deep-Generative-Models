import torch

from ..base import Trainer
from ..utils import nats2bits

class AdversarialTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__(model, train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

        self.dtype = next(self.model.parameters()).dtype
        
        # config optimizor
        self.discriminator_optim = torch.optim.Adam(model.discriminator.parameters(), 
            lr=config['discriminator_lr'], betas=(config['discriminator_beta1'], config['discriminator_beta2']))
        self.generator_optim = torch.optim.Adam(model.generator.parameters(), 
            lr=config['generator_lr'], betas=(config['generator_beta1'], config['generator_beta2']))

        self.fix_z = torch.randn(64, config['latent_dim'], device=self.device, dtype=self.dtype)
    
    def train_step(self):
        for i in range(self.config['n_critic']):
            real = next(self.train_iter)[0].to(self.device)
            fake = self.model.generate(real.shape[0])

            discriminator_loss = self.model.get_discriminator_loss(real, fake)

            self.discriminator_optim.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optim.step()

            if self.config['model'] == 'WGAN':
                with torch.no_grad():
                    for name, value in self.model.discriminator.named_parameters():
                        if 'weight' in name:
                            value.clamp_(-self.config['clamp'], self.config['clamp'])

        # train generator only once
        fake = self.model.generate(real.shape[0])
        generator_loss = self.model.get_generator_loss(fake)

        self.generator_optim.zero_grad()
        generator_loss.backward()
        self.generator_optim.step()

        self.last_discriminator_loss = discriminator_loss.item()
        self.last_generator_loss = generator_loss.item()

    def log_step(self, step):
        print('In Step {}'.format(step))
        print('-' * 15)
        print('Discriminator loss is {0:{1}}'.format(self.last_discriminator_loss, '.3f'))
        print('Generator loss is {0:{1}}'.format(self.last_generator_loss, '.3f'))

        self.writer.add_scalars('loss', {'discriminator' : self.last_discriminator_loss, 
                                    'generator' : self.last_generator_loss}, global_step=step)

        imgs = torch.clamp(self.model.z2x(self.fix_z) / 2 + 0.5, 0, 1)
        self.writer.add_images('fixed_samples', imgs, global_step=step)

        imgs = torch.clamp(self.model.generate(64) / 2 + 0.5, 0, 1)
        self.writer.add_images('samples', imgs, global_step=step)

    def save(self, filename):
        torch.save({
            "discriminator_state_dict" : self.model.discriminator.state_dict(),
            "generator_state_dict" : self.model.generator.state_dict(),
            "discriminator_optimizer_state_dict" : self.discriminator_optim.state_dict(),
            "generator_optimizer_state_dict" : self.generator_optim.state_dict(),
            "config" : self.config,
            "model_parameters" : self.config['model_param'],
            "seed" : self.config['seed'],
        }, filename)

    def restore(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.model = self.model.to(self.device) # make sure model on right device
        self.discriminator_optim.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.generator_optim.load_state_dict(checkpoint['generator_optimizer_state_dict'])