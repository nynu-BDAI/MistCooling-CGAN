import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim=3, output_channels=3):

        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels

        self.fc = nn.Linear(self.input_dim, 256 * 16 * 16)

        self.deconv1 = nn.ConvTranspose2d(256, 1024, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(512)

        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)


        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(64)


        self.conv_final = nn.Conv2d(64, self.output_channels, 3, 1, 1)

        self.activation = nn.ReLU(inplace=True)
        self.output_activation = nn.Sigmoid()

    def forward(self, blowing_ratio, mist_concentration, drop_diameter):

        params = torch.cat((blowing_ratio, mist_concentration, drop_diameter), dim=1)


        x = self.fc(params)
        x = x.view(-1, 256, 16, 16)


        x = self.activation(self.bn1(self.deconv1(x)))
        x = self.activation(self.bn2(self.deconv2(x)))
        x = self.activation(self.bn3(self.deconv3(x)))

        x = self.activation(self.bn4(self.conv1(x)))
        x = self.activation(self.bn5(self.conv2(x)))

        x = self.output_activation(self.conv_final(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):

        super(Discriminator, self).__init__()

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(6, 16, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_linear = nn.Linear(128, 1)

    def forward(self, blowing_ratio, mist_concentration, drop_diameter, image):
        # 参数维度扩展与拼接
        params = torch.cat((blowing_ratio, mist_concentration, drop_diameter), dim=1)
        params = params.unsqueeze(2).unsqueeze(3).expand(-1, -1, image.size(2), image.size(3))

        x = torch.cat((params, image), dim=1)  # 6 通道

        # CNN 特征提取
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.final_linear(x)
        return self.sigmoid(x)