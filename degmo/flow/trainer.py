import torch, torchvision
import numpy as np

from ..base import Trainer
from ..utils import nats2bits

class FlowTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__(model, train_loader, val_loader=val_loader, test_loader=test_loader, config=config)
        
        # config optimizer
        if self.config['model'] == 'RealNVP':
            self.optim = torch.optim.Adam([
                {'params' : [parameter[1] for parameter in model.named_parameters() if not 'scale' in parameter[0]]},
                {'params' : [parameter[1] for parameter in model.named_parameters() if 'scale' in parameter[0]], 'weight_decay' : 5e-5}
            ], lr=config['lr'], betas=(config['beta1'], config['beta2']))
        else:
            self.optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']))

        # Initialize ActNorm
        if config['model']== 'Glow':
            schedule = lambda step: min(step / config['warmup_steps'], 1.0)
            self.optim_schedule = torch.optim.lr_scheduler.LambdaLR(self.optim, schedule)
            # Initialize ActNorm
            init_loader = torch.utils.data.DataLoader(self.train_loader.dataset, batch_size=config['actnorm_batch_size'], shuffle=True)
            init_data = next(iter(init_loader))[0].to(self.device)
            with torch.no_grad():
                model(init_data)
            del init_data, init_loader
            print('ActNorm Initialize complete!')

        # draw fix z
        with torch.no_grad():
            batch = next(self.train_iter)[0].to(self.device)
            z = self.model.x2z(batch)
            self.fix_z = torch.randn(64, *z.shape[1:], device=z.device, dtype=z.dtype)
            self.dim = np.prod(z.shape[1:])
    
    def train_step(self):
        batch = next(self.train_iter)[0].to(self.device)

        _, loss = self.model(batch)

        # temporal solution: discard bad batch
        if torch.isnan(loss) or torch.isinf(loss):
            loss.backward()
            self.optim.zero_grad()
            print("bad batch, skip")
            return
        
        if self.config['model'] == 'Glow':
            self.optim_schedule.step()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.last_train_loss = loss.item()

    def log_step(self, step):
        with torch.no_grad():
            val_loss = self.test_whole(self.val_loader)

            print('In Step {}'.format(step))
            print('-' * 15)
            print('In training set, NLL is {0:{1}} bits / dim'.format(nats2bits(self.last_train_loss) / self.dim, '.3f'))
            print('In validation set, NLL is {0:{1}} bits / dim'.format(nats2bits(val_loss) / self.dim, '.3f'))

            self.writer.add_scalars('NLL', {'train' : nats2bits(self.last_train_loss) / self.dim, 
                                        'val' : nats2bits(val_loss) / self.dim}, global_step=step)

            imgs = torch.clamp(self.model.sample(64), 0, 1)
            self.writer.add_images('samples', imgs, global_step=step)

            imgs = torch.clamp(self.model.z2x(self.fix_z), 0, 1)
            self.writer.add_images('fixed_samples', imgs, global_step=step)

    def test_whole(self, loader):
        with torch.no_grad():
            num = 0
            loss = 0
            for batch in iter(loader):
                num += 1
                batch = batch[0].to(self.device)
                _, _loss = self.model(batch)
                loss += _loss
            loss = loss / num
        return loss.item()

    def save(self, filename):
        test_loss = self.test_whole(self.test_loader)
        torch.save({
            "model_state_dict" : self.model.state_dict(),
            "optimizer_state_dict" : self.optim.state_dict(),
            "test_loss" : nats2bits(test_loss) / self.dim,
            "config" : self.config,
            "model_parameters" : self.config['model_param'],
            "seed" : self.config['seed'],
        }, filename)

    def restore(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device) # make sure model on right device
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])