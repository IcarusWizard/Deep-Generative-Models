from .vae import VAE
from .fvae import FVAE
from .vqvae import VQ_VAE
import torch
from tqdm import tqdm

from ..utils import step_loader, nats2bits

def config_model(args, model_param):
    model_param['network_type'] = args.network_type

    if args.network_type == 'mlp':
        model_param['config'] = {
            "features" : args.features, 
            "hidden_layers" : args.hidden_layers,
        }
    else:
        if args.network_type == 'conv':
            model_param['config'] = {
                "conv_features" : args.conv_features,
                "down_sampling" : args.down_sampling,
                "batchnorm" : args.use_batchnorm,
                "mlp_features" : args.features,
                "mlp_layers" : args.hidden_layers,
            } 
        elif args.network_type == 'fullconv':
            model_param['config'] = {
                "conv_features" : args.conv_features,
                "down_sampling" : args.down_sampling,
                "batchnorm" : args.use_batchnorm,
            } 

        assert len(args.res_layers) == 1 or len(args.res_layers) == args.down_sampling
        if len(args.res_layers) == 1:
            model_param['config']['res_layers'] = args.res_layers * args.down_sampling
        else:
            model_param['config']['res_layers'] = args.res_layers       

    if args.model == 'VAE':
        model_param.update({
            "latent_dim" : args.latent_dim,
            "output_type" : args.output_type,
            "use_mce" : args.use_mce,
            "output_type" : args.output_type,
        })       

        model = VAE(**model_param)   
    elif args.model == 'FVAE':
        model_param.update({
            "latent_dim" : args.latent_dim,
            "flow_features" : args.flow_features,
            "flow_hidden_layers" : args.flow_hidden_layers,
            "flow_num_transformation" : args.flow_num_transformation,
        })        
        model = FVAE(**model_param)
    elif args.model == 'VQ-VAE':
        model_param.update({
            "k" : args.k,
            "d" : args.d,
            "beta" : args.beta,
            "output_type" : args.output_type,
        })     
        model = VQ_VAE(**model_param)
    else:
        raise ValueError('Model {} is not supported!'.format(args.model))

    return model, model_param

def train_vae(model, optim, writer, train_loader, val_loader, test_loader, args):
    device = next(model.parameters()).device
    step = 0
    with tqdm(total=args.steps) as timer:
        for batch in step_loader(train_loader):
            batch = batch[0].to(device)

            loss, info = model(batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % args.log_step == 0:
                val_loss, val_info = test_vae(model, val_loader)

                print('In Step {}'.format(step))
                print('-' * 15)
                print('In training set:')
                print('NELOB is {0:{1}} bits'.format(nats2bits(loss.item()), '.5f'))
                for k in info.keys():
                    print('{0} is {1:{2}} bits'.format(k, nats2bits(info[k].item()), '.5f'))
                print('In validation set:')
                print('NELOB is {0:{1}} bits'.format(nats2bits(val_loss.item()), '.5f'))
                for k in val_info.keys():
                    print('{0} is {1:{2}} bits'.format(k, nats2bits(val_info[k].item()), '.5f'))

                writer.add_scalars('NELOB', {'train' : nats2bits(loss.item()), 
                                             'val' : nats2bits(val_loss.item())}, global_step=step)
                for k in info.keys():
                    writer.add_scalars(k, {'train' : nats2bits(info[k].item()), 
                                           'val' : nats2bits(val_info[k].item())}, global_step=step)

                imgs = torch.clamp(model.sample(64, deterministic=True), 0, 1)
                writer.add_images('samples', imgs, global_step=step)
                input_imgs = batch[:32]
                reconstructions = torch.clamp(model.decode(model.encode(input_imgs)), 0, 1)
                inputs_and_reconstructions = torch.stack([input_imgs, reconstructions], dim=1).view(input_imgs.shape[0] * 2, *input_imgs.shape[1:])
                writer.add_images('inputs_and_reconstructions', inputs_and_reconstructions, global_step=step)

            step = step + 1
            timer.update(1)

            if step >= args.steps:
                break

    test_loss, test_info = test_vae(model, test_loader)
    print('In test set:')
    print('NELOB is {0:{1}} bits'.format(nats2bits(test_loss.item()), '.5f'))
    for k in test_info.keys():
        print('{0} is {1:{2}} bits'.format(k, nats2bits(test_info[k].item()), '.5f'))

    return model, test_loss

def test_vae(model, loader):
    device = next(model.parameters()).device

    with torch.no_grad():
        num = 0
        info = {}
        loss = 0
        for batch in iter(loader):
            num += 1
            batch = batch[0].to(device)
            _loss, _info = model(batch)
            loss += _loss
            for k in _info.keys():
                info[k] = info.get(k, 0) + _info[k]
        loss = loss / num
        for k in info.keys():
            info[k] = info[k] / num

    return loss, info