from .pixelrnn import PixelRNN
from .pixelcnn import PixelCNN

import torch
import numpy as np
from tqdm import tqdm
import time

from ..utils import step_loader, nats2bits

def config_model(args, model_param):
    if args.model == 'PixelRNN':
        model_param.update({
            "mode" : args.mode,
            "features" : args.features,
            "layers" : args.layers,
            "post_features" : args.post_features,
            "bits" : args.bits,
        })
        model = PixelRNN(**model_param)
    elif args.model == 'PixelCNN':
        model_param.update({
            "mode" : args.mode,
            "features" : args.features,
            "layers" : args.layers,
            "bits" : args.bits,
            "filter_size" : args.filter_size,
            "post_features" : args.post_features,
            "bits" : args.bits,
        })
        model = PixelCNN(**model_param)
    else:
        raise ValueError('Model {} is not supported!'.format(args.model))

    return model, model_param

def train_autoregressive(model, optim, writer, train_loader, val_loader, test_loader, args):
    device = next(model.parameters()).device

    step = 0

    with tqdm(total=args.steps) as timer:
        for batch in step_loader(train_loader):
            batch = batch[0].to(device)
            
            loss = model(batch)
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % args.log_step == 0:
                dim = np.prod(batch.shape[1:])
                with torch.no_grad():
                    val_loss = test_autoregressive(model, val_loader)

                    print('In Step {}'.format(step))
                    print('-' * 15)
                    print('In training set, NLL is {0:{1}} bits / dim'.format(nats2bits(loss.item()), '.3f'))
                    print('In validation set, NLL is {0:{1}} bits / dim'.format(nats2bits(val_loss.item()), '.3f'))

                    writer.add_scalars('NLL', {'train' : nats2bits(loss.item()), 
                                                'val' : nats2bits(val_loss.item())}, global_step=step)
            if step % args.generation_step == 0:
                with torch.no_grad():
                    start_time = time.time()
                    imgs = model.sample(64)
                    print("use {} seconds to generate 64 images".format(time.time() - start_time))
                    writer.add_images('samples', imgs, global_step=step)

            step = step + 1
            timer.update(1)

            if step >= args.steps:
                break

    test_loss = test_autoregressive(model, test_loader)

    print('In test set, NLL is {0:{1}} bits / dim'.format(nats2bits(test_loss.item()), '.3f'))

    return model, test_loss

def test_autoregressive(model, loader):
    device = next(model.parameters()).device
    with torch.no_grad():
        num = 0
        loss = 0
        for batch in iter(loader):
            num += 1
            batch = batch[0].to(device)
            _loss = model(batch)
            loss += _loss
        loss = loss / num
    return loss