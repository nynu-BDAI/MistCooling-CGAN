# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import argparse
import time
import warnings
import csv
from modules.model import Generator, Discriminator
from Dataset import CustomDataset
from utils.config_utils import load_config
from utils.Timer import Timer
from utils.AverageMeter import AverageMeter
from utils.fs_utils import create_folder

warnings.filterwarnings("ignore", category=UserWarning)

create_folder('checkpoints')
folderPath = 'checkpoints/session_' + Timer.timeFilenameString() + '/'
create_folder(folderPath)
create_folder('log')
logPath = 'log/log_' + Timer.timeFilenameString()


def append_line_to_log(line='\n'):
    with open(logPath, 'a') as f:
        f.write(line + '\n')


def parse_cli():
    parser = argparse.ArgumentParser(description='PyTorch CGAN Training')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr_G', type=float, default=1e-5)
    parser.add_argument('--lr_D', type=float, default=1e-5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-08)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--train_dir', default="", type=str)
    parser.add_argument('--val_dir', default="", type=str)
    return parser.parse_args()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def train(epoch, generator, discriminator, optimizer_G, optimizer_D, criterion_GAN, criterion_L1, lambda_L1, loader,
          device, log_callback, log_interval, scheduler_G, scheduler_D, train_losses_G, train_losses_D):
    generator.train()
    discriminator.train()
    total_loss_G, total_loss_D = 0, 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()


    for batch_idx, (target_image, br, mc, dd) in enumerate(loader):
        data_time.update(time.time() - end)

        # 将数据移至 GPU
        target_image = target_image.to(device)
        br = br.to(device)
        mc = mc.to(device)
        dd = dd.to(device)

        # --- 训练判别器 ---
        optimizer_D.zero_grad()


        fake_image = generator(br, mc, dd)

        real_pred = discriminator(br, mc, dd, target_image)
        fake_pred = discriminator(br, mc, dd, fake_image.detach())

        loss_D = (criterion_GAN(real_pred, torch.full_like(real_pred, 0.9)) +
                  criterion_GAN(fake_pred, torch.zeros_like(fake_pred))) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # --- 训练生成器 ---
        optimizer_G.zero_grad()

        fake_pred_g = discriminator(br, mc, dd, fake_image)

        loss_G_GAN = criterion_GAN(fake_pred_g, torch.full_like(fake_pred_g, 0.9))
        loss_G_L1 = criterion_L1(fake_image, target_image) * lambda_L1

        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optimizer_G.step()

        total_loss_G += loss_G.item()
        total_loss_D += loss_D.item()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(loader.dataset)}/{len(loader.dataset)} ({100. * batch_idx / len(loader):.0f}%)]\tLoss D: {loss_D.item():.6f}\tLoss G: {loss_G.item():.6f}')

    avg_loss_G = total_loss_G / len(loader)
    avg_loss_D = total_loss_D / len(loader)
    train_losses_G.append(avg_loss_G)
    train_losses_D.append(avg_loss_D)
    scheduler_G.step(avg_loss_G)
    scheduler_D.step(avg_loss_D)


def validate(generator, discriminator, criterion_GAN, criterion_L1, lambda_L1, loader, device, log_callback):
    generator.eval()
    discriminator.eval()
    val_loss_G, val_loss_D = 0, 0

    with torch.no_grad():

        for batch_idx, (target_image, br, mc, dd) in enumerate(loader):
            target_image = target_image.to(device)
            br = br.to(device)
            mc = mc.to(device)
            dd = dd.to(device)


            fake_image = generator(br, mc, dd)


            output_real = discriminator(br, mc, dd, target_image)
            output_fake = discriminator(br, mc, dd, fake_image)

            real_label = torch.full_like(output_real, 0.9, device=device)
            fake_label = torch.zeros_like(output_fake, device=device)

            loss_D = (criterion_GAN(output_real, real_label) + criterion_GAN(output_fake, fake_label)) * 0.5
            loss_G = criterion_GAN(output_fake, real_label) + criterion_L1(fake_image, target_image) * lambda_L1

            val_loss_G += loss_G.item()
            val_loss_D += loss_D.item()

    avg_val_loss_G = val_loss_G / len(loader)
    avg_val_loss_D = val_loss_D / len(loader)
    print(f'Validation Loss D: {avg_val_loss_D:.6f}\tLoss G: {avg_val_loss_G:.6f}')
    return avg_val_loss_G, avg_val_loss_D


def main():
    args = parse_cli()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    train_dataset = CustomDataset(args.train_dir, is_train=True, output_size=128)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset),
                              num_workers=args.workers, pin_memory=True)

    val_dataset = CustomDataset(args.val_dir, is_train=False, output_size=128)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2), eps=args.epsilon)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2), eps=args.epsilon)

    scheduler_G = lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.7, patience=2, verbose=True)
    scheduler_D = lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.7, patience=2, verbose=True)

    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    lambda_L1 = 100

    csv_log_path = os.path.join(folderPath, 'training_log.csv')
    with open(csv_log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train_Loss_G', 'Train_Loss_D', 'Val_Loss_G', 'Val_Loss_D'])

    best_val_loss = np.inf
    train_losses_G, train_losses_D = [], []

    for epoch in range(1, args.epochs + 1):
        train(epoch, generator, discriminator, optimizer_G, optimizer_D, criterion_GAN, criterion_L1, lambda_L1,
              train_loader, device, append_line_to_log, args.log_interval, scheduler_G, scheduler_D, train_losses_G,
              train_losses_D)
        val_loss_G, val_loss_D = validate(generator, discriminator, criterion_GAN, criterion_L1, lambda_L1, val_loader,
                                          device, append_line_to_log)

        with open(csv_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_losses_G[-1], train_losses_D[-1], val_loss_G, val_loss_D])

        if val_loss_G < best_val_loss - 1e-4:
            best_val_loss = val_loss_G
            torch.save({'model_state_dict': generator.state_dict()}, os.path.join(folderPath, f'best_model_G.pth'))
            torch.save({'model_state_dict': discriminator.state_dict()}, os.path.join(folderPath, f'best_model_D.pth'))
            print(f'Saved best model at epoch {epoch}')


if __name__ == '__main__':
    main()