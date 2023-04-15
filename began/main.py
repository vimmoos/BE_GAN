import wandb
import random
import torch
from began.dataset import CelebA
from began.models import Discriminator, Decoder
from began.train import train
import torch.optim as optim


# # start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="began",
    # track hyperparameters and run metadata
    config=dict(
        manual_seed=84,
        data_path="data/32_32_crop/",
        img_size=32,
        workers_dl=2,
        batch_size=64,
        learning_rate=0.0001,
        device="cpu",
        beta1=0.5,
        gamma=0.5,
        lambda_k=0.001,
        niter=30000,
    ),
)

# opt = wandb.config

manual_seed = 84
data_path = "data/32_32_crop/"
img_size = 32
workers_dl = 2
batch_size = 64
lr = 0.0001
beta1 = 0.5
gamma = 0.5
lambda_k = 0.001
niter = 30000
device = (
    "cpu"  # torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
)
print("Random Seed: ", manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# if cuda:
#     torch.cuda.manual_seed_all(manual_seed)


dataset = CelebA(data_path, img_size)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers_dl,
)


criterion = torch.nn.L1Loss()
netD = Discriminator()
netG = Decoder()

wandb.watch(netG)
wandb.watch(netD)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

train(
    max_iter=niter,
    dataloader=dataloader,
    netD=netD,
    optimizerD=optimizerD,
    optimizerG=optimizerG,
    netG=netG,
    batch_size=batch_size,
    device=device,
    criterion=criterion,
    gamma=gamma,
    lambda_k=lambda_k,
    logger=wandb.log,
    init_lr=lr,
    lr_decay_iter=3000,
    save_step=10000,
)


torch.save(netD.state_dict(), f"disc_dict_{manual_seed}")
torch.save(netD, f"disc_{manual_seed}")
torch.save(netG.state_dict(), f"gen_dict_{manual_seed}")
torch.save(netG, f"gen_{manual_seed}")
