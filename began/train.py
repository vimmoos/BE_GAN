from tqdm import trange
import torch
from typing import Callable
import numpy as np
import os


def train(
    dataloader: torch.utils.data.DataLoader,
    netD: torch.nn.Module,
    optimizerD: torch.optim,
    optimizerG: torch.optim,
    netG: torch.nn.Module,
    criterion: Callable,
    logger: Callable,
    max_iter: int,
    batch_size: int,
    device: str,
    gamma: float,
    lambda_k: float,
    lr_step: int,
    lr_gamma: float,
    save_step: int,
    outfolder: str = "models",
):
    k_t = 0
    loader = iter(dataloader)
    netD.train()
    netG.train()
    schedG = torch.optim.lr_scheduler.StepLR(optimizerG, lr_step, lr_gamma)
    schedD = torch.optim.lr_scheduler.StepLR(optimizerD, lr_step, lr_gamma)

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    for step in trange(0, max_iter):
        try:
            real = next(loader)
        except StopIteration:
            loader = iter(dataloader)
            real = next(loader)

        netD.zero_grad()
        netG.zero_grad()

        noise = torch.randn(batch_size, 64, device=device)
        fake = netG(noise)

        r_recon = netD(real)
        f_recon = netD(fake.detach())

        err_real = criterion(r_recon, real)
        err_fake = criterion(f_recon, fake.detach())

        errD = err_real - k_t * err_fake
        errD.backward()
        optimizerD.step()

        netG.zero_grad()
        fake = netG(noise)
        fake_recons = netD(fake)
        errG = criterion(fake_recons, fake)
        errG.backward()
        optimizerG.step()

        balance = (gamma * err_real - err_fake).item()
        k_t = min(max(k_t + lambda_k * balance, 0), 1)
        measure = err_real.item() + np.abs(balance)

        logger(
            {
                "loss_d": errD.item(),
                "loss_g": errG.item(),
                "k_t": k_t,
                "lr": optimizerD.param_groups[0]["lr"],
                "measure": measure,
            }
        )
        schedD.step()
        schedG.step()
        if (step + 1) % save_step == 0 and step != 0:
            torch.save(
                {
                    "epoch": step,
                    "generator_state_dict": netG.state_dict(),
                    "optimizerG_state_dict": optimizerG.state_dict(),
                    "loss_generator": errG,
                    "discriminator_state_dict": netD.state_dict(),
                    "optimizerD_state_dict": optimizerD.state_dict(),
                    "loss_discriminator": errD,
                },
                f"{outfolder}/BEGAN_{step}.pth",
            )

    return netD, netG
