import time
import torch
import numpy as np

from ..base import Trainer
from ..utils import nats2bits

class AutoregressiveTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__(model, train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

        # config optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], 
                                      betas=(self.config['beta1'], self.config['beta2']))

        batch = next(self.train_iter)[0]
        self.dim = np.prod(batch.shape[1:])
    
    def train_step(self):
        batch = next(self.train_iter)[0].to(self.device)

        loss = self.model(batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.last_train_loss = loss.item()

    def log_step(self, step):
        with torch.no_grad():
            val_loss = self.test_whole(self.val_loader)

            print('In Step {}'.format(step))
            print('-' * 15)
            print('In training set, NLL is {0:{1}} bits / dim'.format(nats2bits(self.last_train_loss), '.3f'))
            print('In validation set, NLL is {0:{1}} bits / dim'.format(nats2bits(val_loss), '.3f'))

            self.writer.add_scalars('NLL', {'train' : nats2bits(self.last_train_loss), 
                                        'val' : nats2bits(val_loss)}, global_step=step)
        if step % self.config['generation_step'] == 0:
            with torch.no_grad():
                start_time = time.time()
                imgs = self.model.sample(64)
                print("use {} seconds to generate 64 images".format(time.time() - start_time))
                self.writer.add_images('samples', imgs, global_step=step)

    def test_whole(self, loader):
        with torch.no_grad():
            num = 0
            loss = 0
            for batch in iter(loader):
                num += 1
                batch = batch[0].to(self.device)
                _loss = self.model(batch)
                loss += _loss
            loss = loss / num
        return loss.item()

    def save(self, filename):
        test_loss = self.test_whole(self.test_loader)
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