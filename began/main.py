import wandb
import random
import torch
from began.dataset import CelebA
from began.models.standard import Discriminator, Decoder
from began.models.skip import SkipDiscriminator, SkipDecoder
from began.train import train
import torch.optim as optim

# import argparse


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="BEGAN network")


def main():
    wandb.init(
        # set the wandb project where this run will be logged
        project="began",
        # track hyperparameters and run metadata
        config=dict(
            # settings
            device="cuda",  #
            manual_seed=84,  #
            save_step=2000,  #
            # data
            data_path="data/32_32_crop/",  #
            img_size=32,  #
            workers_dl=10,  #
            batch_size=64,  #
            # optimizers
            lr=0.0001,  #
            beta1=0.5,  #
            lr_step=5000,  #
            lr_gamma=0.95,  #
            # network
            gamma=0.5,  #
            lambda_k=0.001,  #
            skip=True,  #
            n_filters=64,
            # training
            max_iter=20000,  #
        ),
    )
    conf = wandb.config
    print(conf)

    print("Random Seed: ", conf["manual_seed"])
    random.seed(conf["manual_seed"])
    torch.manual_seed(conf["manual_seed"])

    if conf["device"] != "cpu":
        torch.cuda.manual_seed_all(conf["manual_seed"])

    dataset = CelebA(conf["data_path"], conf["img_size"], load_all=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["workers_dl"],
    )

    criterion = torch.nn.L1Loss()
    netD = (
        Discriminator(n_filters=conf["n_filters"])
        if conf["skip"] == "false"
        else SkipDiscriminator(n_filters=conf["n_filters"])
    )
    netG = (
        Decoder(n_filters=conf["n_filters"])
        if conf["skip"] == "false"
        else SkipDecoder(n_filters=conf["n_filters"])
    )

    netD = torch.jit.script(netD)
    netG = torch.jit.script(netG)

    print(netD)

    wandb.watch(netG)
    wandb.watch(netD)

    optimizerD = optim.Adam(
        netD.parameters(), lr=conf["lr"], betas=(conf["beta1"], 0.999)
    )
    optimizerG = optim.Adam(
        netG.parameters(), lr=conf["lr"], betas=(conf["beta1"], 0.999)
    )

    train(
        # settings
        device=conf["device"],
        max_iter=conf["max_iter"],
        logger=wandb.log,
        # data
        dataloader=dataloader,
        batch_size=conf["batch_size"],
        # network
        netD=netD,
        netG=netG,
        optimizerD=optimizerD,
        optimizerG=optimizerG,
        criterion=criterion,
        gamma=conf["gamma"],
        lambda_k=conf["lambda_k"],
        lr_step=conf["lr_step"],
        lr_gamma=conf["lr_gamma"],
        save_step=conf["save_step"],
        outfolder=f"models/{'_'.join(str(k) + '_' + str(v) for k,v in conf.items() if k != 'data_path')}",
    )


if __name__ == "__main__":
    main()
