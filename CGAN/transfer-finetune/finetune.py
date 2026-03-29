import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from modules.model import Generator, Discriminator
from Dataset import CustomDataset
from utils.Timer import Timer
from utils.fs_utils import create_folder

# ==============================================================================
# 配置区域
# ==============================================================================
PRETRAINED_G_PATH = ""
PRETRAINED_D_PATH = PRETRAINED_G_PATH.replace("best_model_G", "best_model_D")

SOURCE_TRAIN_CSV = ""
SOURCE_TEST_CSV = "/"
DATASET_ROOT_DIR = "/"

BATCH_SIZE = 8
EPOCHS = 100
LR_FINETUNE = 1e-4  # 微调通常使用比从零训练稍小的学习率
LAMBDA_L1 = 100


# ==============================================================================

def prepare_dataset_files():
    target_train = os.path.join(DATASET_ROOT_DIR, 'chairs.train.class.csv')
    target_valid = os.path.join(DATASET_ROOT_DIR, 'chairs.valid.class.csv')
    if os.path.exists(SOURCE_TRAIN_CSV): shutil.copy(SOURCE_TRAIN_CSV, target_train)
    if os.path.exists(SOURCE_TEST_CSV): shutil.copy(SOURCE_TEST_CSV, target_valid)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prepare_dataset_files()

    # 1. 加载数据
    train_loader = DataLoader(CustomDataset(DATASET_ROOT_DIR, is_train=True, output_size=IMAGE_SIZE),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CustomDataset(DATASET_ROOT_DIR, is_train=False, output_size=IMAGE_SIZE),
                            batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型并加载预训练权重
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    print(f"正在加载预训练权重...")
    ckpt_G = torch.load(PRETRAINED_G_PATH, map_location=device)
    generator.load_state_dict(ckpt_G['model_state_dict'] if 'model_state_dict' in ckpt_G else ckpt_G)

    ckpt_D = torch.load(PRETRAINED_D_PATH, map_location=device)
    discriminator.load_state_dict(ckpt_D['model_state_dict'] if 'model_state_dict' in ckpt_D else ckpt_D)

    # 3. 核心迁移学习策略：冻结层
    # 策略：冻结生成器中的所有反卷积层(deconv)，只训练全连接层(fc)
    # 这样模型会保留生成流场细节的能力，只调整物理参数的对应关系
    print("--- 冻结配置 ---")
    for name, param in generator.named_parameters():
        if "deconv" in name:
            param.requires_grad = False
            print(f"[已冻结] {name}")
        else:
            param.requires_grad = True
            print(f"[待微调] {name}")

    # 4. 定义优化器 (注意：只传入 requires_grad=True 的参数)
    optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=LR_FINETUNE)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_FINETUNE * 0.1)  # 判别器通常设得极低

    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()

    # 5. 训练循环
    folderPath = 'checkpoints/finetune_frozen_' + Timer.timeFilenameString() + '/'
    create_folder(folderPath)
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        generator.train()
        train_l1_loss = 0

        for i, (target_image, br, mc, dd) in enumerate(train_loader):
            target_image, br, mc, dd = target_image.to(device), br.to(device), mc.to(device), dd.to(device)

            # 更新 D
            optimizer_D.zero_grad()
            fake_image = generator(br, mc, dd)
            loss_D = (criterion_GAN(discriminator(br, mc, dd, target_image), torch.ones_like(br, device=device)) +
                      criterion_GAN(discriminator(br, mc, dd, fake_image.detach()),
                                    torch.zeros_like(br, device=device))) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # 更新 G
            optimizer_G.zero_grad()
            pred_fake = discriminator(br, mc, dd, fake_image)
            g_gan = criterion_GAN(pred_fake, torch.ones_like(pred_fake, device=device))
            g_l1 = criterion_L1(fake_image, target_image) * LAMBDA_L1

            (g_gan + g_l1).backward()
            optimizer_G.step()
            train_l1_loss += g_l1.item()

        # 验证
        generator.eval()
        val_l1 = 0
        with torch.no_grad():
            for ti, b, m, d in val_loader:
                fi = generator(b.to(device), m.to(device), d.to(device))
                val_l1 += criterion_L1(fi, ti.to(device)).item()

        avg_val = val_l1 / len(val_loader)
        print(f"Epoch [{epoch}/{EPOCHS}] Train L1: {train_l1_loss / len(train_loader):.5f} | Val L1: {avg_val:.5f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(generator.state_dict(), os.path.join(folderPath, 'best_model_G.pth'))
            print("★ 发现更优模型，已保存。")


if __name__ == '__main__':
    main()