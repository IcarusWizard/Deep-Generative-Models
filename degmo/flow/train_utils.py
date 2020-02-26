from .nice import NICE
from .glow import GLOW
from .realnvp import RealNVP2D

import torch
import numpy as np
from tqdm import tqdm

from ..utils import step_loader, nats2bits

def config_model(args, model_param):
    model_param.update({
        "features" : args.features,
        "bits" : args.bits,
    })
    if args.model == 'NICE':
        model_param.update({
            "hidden_layers" : args.blocks,
        })
        model = NICE(**model_param)        
    elif args.model == 'RealNVP':
        model_param.update({
            "hidden_blocks" : args.blocks,
            "down_sampling" : args.down_sampling,
        })
        model = RealNVP2D(**model_param)
    elif args.model == 'Glow':
        model_param.update({
            "K" : args.K,
            "L" : args.L,
            "use_lu" : not args.nolu,
            "coupling" : args.coupling,
        })
        model = GLOW(**model_param)
    else:
        raise ValueError('Model {} is not supported!'.format(args.model))

    return model, model_param

def train_flow(model,  writer, train_loader, val_loader, test_loader, args):
    device = next(model.parameters()).device

    # config optimizer
    if args.model == 'RealNVP':
        optim = torch.optim.Adam([
            {'params' : [parameter[1] for parameter in model.named_parameters() if not 'scale' in parameter[0]]},
            {'params' : [parameter[1] for parameter in model.named_parameters() if 'scale' in parameter[0]], 'weight_decay' : 5e-5}
        ], lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    if args.model == 'Glow':
        schedule = lambda step: min(step / args.warmup_steps, 1.0)
        optim_schedule = torch.optim.lr_scheduler.LambdaLR(optim, schedule)
        # Initialize ActNorm
        init_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=args.actnorm_batch_size, shuffle=True)
        init_data = next(iter(init_loader))[0].to(device)
        with torch.no_grad():
            model(init_data)
        del init_data, init_loader
        print('ActNorm Initialize complete!')

    step = 0

    with tqdm(total=args.steps) as timer:
        for batch in step_loader(train_loader):
            batch = batch[0].to(device)
            
            _, loss = model(batch)

            # temporal solution: discard bad batch
            if torch.isnan(loss) or torch.isinf(loss):
                loss.backward()
                optim.zero_grad()
                print("bad batch, skip")
                continue
            
            if args.model == 'Glow':
                optim_schedule.step()
            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % args.log_step == 0:
                dim = np.prod(batch.shape[1:])
                with torch.no_grad():
                    val_loss = test_flow(model, val_loader)

                    print('In Step {}'.format(step))
                    print('-' * 15)
                    print('In training set, NLL is {0:{1}} bits / dim'.format(nats2bits(loss.item()) / dim, '.3f'))
                    print('In validation set, NLL is {0:{1}} bits / dim'.format(nats2bits(val_loss.item()) / dim, '.3f'))

                    writer.add_scalars('NLL', {'train' : nats2bits(loss.item()) / dim, 
                                                'val' : nats2bits(val_loss.item()) / dim}, global_step=step)
                    imgs = torch.clamp(model.sample(64), 0, 1)
                    writer.add_images('samples', imgs, global_step=step)

            step = step + 1
            timer.update(1)

            if step >= args.steps:
                break

    test_loss = test_flow(model, test_loader)

    print('In test set, NLL is {0:{1}} bits / dim'.format(nats2bits(test_loss.item()) / dim, '.3f'))

    return model, optim, test_loss

def test_flow(model, loader):
    device = next(model.parameters()).device
    with torch.no_grad():
        num = 0
        loss = 0
        for batch in iter(loader):
            num += 1
            batch = batch[0].to(device)
            _, _loss = model(batch)
            loss += _loss
        loss = loss / num
    return loss