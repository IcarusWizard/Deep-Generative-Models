import torch

from ..base import Trainer
from ..utils import nats2bits

class VAETrainer(Trainer):
    def __init__(self, model, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__(model, train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

        # config optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], 
                                      betas=(self.config['beta1'], self.config['beta2']))
    
    def train_step(self):
        batch = next(self.train_iter)[0].to(self.device)

        loss, info = self.model(batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.last_train_loss = loss.item()
        self.last_train_info = info

    def log_step(self, step):
        val_loss, val_info = self.test_whole(self.val_loader)

        print('In Step {}'.format(step))
        print('-' * 15)
        print('In training set:')
        print('NELBO is {0:{1}} bits'.format(nats2bits(self.last_train_loss), '.5f'))
        for k in self.last_train_info.keys():
            print('{0} is {1:{2}} bits'.format(k, nats2bits(self.last_train_info[k]), '.5f'))
        print('In validation set:')
        print('NELBO is {0:{1}} bits'.format(nats2bits(val_loss), '.5f'))
        for k in val_info.keys():
            print('{0} is {1:{2}} bits'.format(k, nats2bits(val_info[k]), '.5f'))

        self.writer.add_scalars('NELBO', {'train' : nats2bits(self.last_train_loss), 
                                        'val' : nats2bits(val_loss)}, global_step=step)
        for k in self.last_train_info.keys():
            self.writer.add_scalars(k, {'train' : nats2bits(self.last_train_info[k]), 
                                    'val' : nats2bits(val_info[k])}, global_step=step)
        
        with torch.no_grad():
            imgs = torch.clamp(self.model.sample(64, deterministic=True), 0, 1)
            self.writer.add_images('samples', imgs, global_step=step)
            input_imgs = batch = next(self.train_iter)[0].to(self.device)[:32]
            reconstructions = torch.clamp(self.model.decode(self.model.encode(input_imgs)), 0, 1)
            inputs_and_reconstructions = torch.stack([input_imgs, reconstructions], dim=1).view(input_imgs.shape[0] * 2, *input_imgs.shape[1:])
            self.writer.add_images('inputs_and_reconstructions', inputs_and_reconstructions, global_step=step)

    def test_whole(self, loader):
        with torch.no_grad():
            num = 0
            info = {}
            loss = 0
            for batch in iter(loader):
                num += 1
                batch = batch[0].to(self.device)
                _loss, _info = self.model(batch)
                loss += _loss
                for k in _info.keys():
                    info[k] = info.get(k, 0) + _info[k]
            loss = loss / num
            for k in info.keys():
                info[k] = info[k] / num

        return loss.item(), info

    def save(self, filename):
        test_loss, _ = self.test_whole(self.test_loader)
        torch.save({
            "model_state_dict" : self.model.state_dict(),
            "optimizer_state_dict" : self.optim.state_dict(),
            "test_loss" : nats2bits(test_loss),
            "config" : self.config,
            "model_parameters" : self.config['model_param'],
            "seed" : self.config['seed'],
        }, filename)

    def restore(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device) # make sure model on right device
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])