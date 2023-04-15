from tqdm import trange
import torch
from typing import Callable
import numpy as np


def adjust_learning_rate(optimizer, lr, lr_decay_iter, niter):
    lr = lr * (0.95 ** (niter // lr_decay_iter))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


def train(
    max_iter: int,
    dataloader: torch.utils.data.DataLoader,
    netD: torch.nn.Module,
    optimizerD: torch.optim,
    optimizerG: torch.optim,
    netG: torch.nn.Module,
    batch_size: int,
    device: str,
    criterion: Callable,
    gamma: float,
    lambda_k: float,
    logger: Callable,
    init_lr: float,
    lr_decay_iter: float,
    save_step: int,
    outfolder: str = "models",
):
    k_t = 0
    loader = iter(dataloader)
    for step in trange(0, max_iter):
        try:
            x = next(loader)
        except StopIteration:
            loader = iter(dataloader)
            x = next(loader)

        netD.zero_grad()
        netG.zero_grad()
        real = x

        noise = torch.randn(batch_size, 64, device=device)
        fake = netG(noise)

        # Forward pass real batch through D
        r_recon = netD(real)
        f_recon = netD(fake.detach())

        # Calculate loss on all-real batch
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
        optimizerD = adjust_learning_rate(
            optimizerD, init_lr, lr_decay_iter, step
        )
        optimizerG = adjust_learning_rate(
            optimizerG, init_lr, lr_decay_iter, step
        )
        if step % save_step == 0 and step != 0:
            torch.save(
                netG.state_dict(),
                f"{outfolder}/netG_{step}.pth",
            )
            torch.save(netD.state_dict(), f"{outfolder}/netG_{step}.pth")

    return netD, netG
