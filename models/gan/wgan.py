import torch, torchvision
from torch import nn
from torch.functional import F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class upsample_conv2d(torch.nn.Module):
    def __init__(self, input_dim, output_dim=256, filter_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_dim, output_dim, filter_size, padding=(filter_size // 2))

    def forward(self, x):
        _x = torch.cat([x, x, x, x], dim=1)
        _x = F.pixel_shuffle(_x, 2)
        return self.conv(_x)

class downsample_conv2d(torch.nn.Module):
    def __init__(self, input_dim, output_dim=256, filter_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_dim, output_dim, filter_size, padding=(filter_size // 2))

    def forward(self, x):
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C*4, H//2, W//2)
        x = torch.mean(x.view(B, 4, C, H//2, W//2), dim=1)
        return self.conv(x)

class ResBlockUp(torch.nn.Module):
    def __init__(self, input_dim, output_dim, filter_size):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(input_dim)
        self.conv = torch.nn.Conv2d(input_dim, output_dim, filter_size, padding=(filter_size // 2))
        self.bn2 = torch.nn.BatchNorm2d(output_dim)

        self.residual = upsample_conv2d(output_dim, output_dim, filter_size)
        self.shortcut = upsample_conv2d(input_dim, output_dim, 1)

    def forward(self, x):
        _x = F.relu(self.bn1(x))
        _x = F.relu(self.bn2(self.conv(_x)))

        return self.residual(_x) + self.shortcut(x)

class ResBlockDown(torch.nn.Module):
    def __init__(self, input_dim, output_dim, filter_size):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_dim, output_dim, filter_size, padding=(filter_size // 2))

        self.residual = downsample_conv2d(output_dim, output_dim, filter_size)
        self.shortcut = downsample_conv2d(input_dim, output_dim, 1)

    def forward(self, x):
        _x = F.relu(x)
        _x = F.relu(self.conv(_x))

        return self.residual(_x) + self.shortcut(x)

class ResBlockGenerator(torch.nn.Module):
    def __init__(self, filter_num=128, filter_size=3):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(filter_num)
        self.conv = torch.nn.Conv2d(filter_num, filter_num, filter_size, padding=(filter_size // 2))
        self.bn2 = torch.nn.BatchNorm2d(filter_num)

    def forward(self, x):
        _x = F.relu(self.bn1(x))
        _x = F.relu(self.bn2(self.conv(_x)))

        return x + _x

class ResBlockDiscriminator(torch.nn.Module):
    def __init__(self, filter_num=128, filter_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(filter_num, filter_num, filter_size, padding=(filter_size // 2))

    def forward(self, x):
        _x = F.relu(x)
        _x = F.relu(self.conv(_x))

        return x + _x

class Generator(torch.nn.Module):
    def __init__(self, hidden_dim=128, filter_num=256, filter_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.filter_num = filter_num
        self.dense = torch.nn.Linear(hidden_dim, 4 * 4 * filter_num)
        self.upblocks = torch.nn.ModuleList([ResBlockUp(filter_num, filter_num, filter_size) for i in range(3)])

        self.output_bn = torch.nn.BatchNorm2d(filter_num)
        self.output_conv = torch.nn.Conv2d(filter_num, 3, 3, padding=1)

    def forward(self, sample_num):
        device = next(self.parameters()).device
        z = torch.randn(sample_num, self.hidden_dim, dtype=torch.float32, device=device)

        space_z = self.dense(z).view(-1, self.filter_num, 4, 4)

        for upblock in self.upblocks:
            space_z = upblock(space_z)

        out = F.relu(self.output_bn(space_z))
        return torch.tanh(self.output_conv(out))

class Discriminator(torch.nn.Module):
    def __init__(self, filter_num=128, filter_size=3):
        super().__init__()
        self.downblocks = torch.nn.ModuleList([ResBlockDown(3, filter_num, filter_size), ResBlockDown(filter_num, filter_num, filter_size)])
        self.resblocks = torch.nn.ModuleList([ResBlockDiscriminator(filter_num, filter_size) for _ in range(2)])

        self.dense = torch.nn.Linear(filter_num, 1)

    def forward(self, x):
        _x = x

        for downblock in self.downblocks:
            _x = downblock(_x)

        for resblock in self.resblocks:
            _x = resblock(_x)

        _x = F.relu(_x)
        _x = torch.sum(torch.sum(_x, dim=-1), dim=-1)

        return self.dense(_x)

class WGAN(torch.nn.Module):
    def __init__(self, dataset='cifar10', batch_size=128, train_steps=100000,
                hidden_dim=128, generator_filter=256, discriminator_filter=128, 
                n_critic=5, lambdav=10,
                lr=2e-4, beta1=0, beta2=0.9,
                model_name='task1'):
        super().__init__()
        if dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.dataset = torchvision.datasets.CIFAR10('./dataset', download=True, transform=transform)
        else:
            raise NameError('dataset {} is not supported!'.format(dataset))
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.n_critic = n_critic
        self.lambdav = lambdav
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.generator = Generator(hidden_dim, generator_filter, 3)
        self.discriminator = Discriminator(discriminator_filter, 3)

        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.generator_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        schedule = lambda step: (self.train_steps - step) / self.train_steps
        self.discriminator_opt_schedule = torch.optim.lr_scheduler.LambdaLR(self.discriminator_opt, schedule)
        self.generator_opt_schedule = torch.optim.lr_scheduler.LambdaLR(self.generator_opt, schedule)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size, num_workers=2)

        self.writer = SummaryWriter('./logs/{}/'.format(model_name))

    def generate(self, sample_num):
        return self.generator(sample_num)

    def train_model(self, steps):
        step = 0
        device = next(self.parameters()).device
        data = iter(self.dataloader)
        while True:
            # train the discriminator first
            for i in range(self.n_critic):
                # get real data
                try:
                    x, _ = next(data)
                except:
                    data = iter(self.dataloader)
                    x, _ = next(data)
                if not x.shape[0] == self.batch_size: 
                    data = iter(self.dataloader)
                    x, _ = next(data)
                x = x.to(device)                

                # get fake data
                x_tilde = self.generator(self.batch_size)

                # mix data
                epsilon = torch.rand(self.batch_size, 1, 1, 1, dtype=torch.float32, device=device)
                x_hat = epsilon * x + (1 - epsilon) * x_tilde

                # compute loss
                prob_x_tilde = self.discriminator(x_tilde)
                prob_x = self.discriminator(x)

                prob_x_hat = self.discriminator(x_hat)
                grad_x_hat = torch.autograd.grad(prob_x_hat, x_hat, torch.ones(*prob_x_hat.shape, device=device), 
                                                create_graph=True, retain_graph=True)[0].view(self.batch_size, -1)

                D_x_tilde = torch.mean(prob_x_tilde)
                D_x = torch.mean(prob_x)
                Lipshitz = torch.mean((torch.sqrt(torch.sum(grad_x_hat ** 2, dim=1)) - 1) ** 2)
                
                loss = D_x_tilde - D_x + self.lambdav * Lipshitz

                loss_value = loss.item()

                # back propagation
                self.discriminator_opt.zero_grad()
                loss.backward()
                self.discriminator_opt.step()
            
            # then train the generator
            x_tilde = self.generator(self.batch_size)
            loss = - torch.mean(self.discriminator(x_tilde))
            self.generator_opt.zero_grad()
            loss.backward()
            self.generator_opt.step()

            # log
            self.discriminator_opt_schedule.step()
            self.generator_opt_schedule.step()

            if step % 200 == 0:
                self.writer.add_images('generation', x_tilde / 2 + 0.5, global_step=step)
                self.writer.add_scalars('Discriminator_Distance', {'X_tilde' : D_x_tilde.item(), 'X' : D_x.item()}, global_step=step)
                self.writer.add_scalar('Lipshitz', Lipshitz.item(), global_step=step)
                self.writer.add_scalar('loss', loss_value, global_step=step)
                print('In step {}, D_x is {}, D_x_tilde is {}, Lipshitz is {}, total loss is {}'.format(
                    step, D_x.item(), D_x_tilde.item(), Lipshitz.item(), loss_value))

            step += 1
            if step >= steps:
                break