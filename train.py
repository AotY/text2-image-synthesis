#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Text2ImageDataset
from modul.gan import Generator, Discriminator

lr = 0.0005
batch_size = 64
epochs = 200


device = torch.device('cuda')

flowers_dataset_path = './data/flowers.hdf5'
split = 0
data_set = Text2ImageDataset(flowers_dataset_path, split=split)

generator = nn.DataParallel(Generator(
    image_size=64,
    num_channels=3,
    noise_size=100,
    embedding_size=1024,
    projected_embedding_size=128,
    ngf=64
))

discriminator = nn.DataParallel(Discriminator(
    image_size=64,
    num_channels=3,
    embedding_size=1024,
    projected_embedding_size=128,
    ndf=64,
    b_size=128,
    c_size=16
))

optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas(0,9, 0.999))

def train():
    criterion = nn.BCELoss()
    l2_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    iter = 0

    for epoch in range(epochs):
        for sample in data_set:
            iter += 1

            right_images = sample['right_images'].float().to(device)
            right_embedded = sample['right_embedded'].float().to(device)
            wrong_images = sample['wrong_images'].float().to(device)
            print('right_images shape: {}'.format(right_images.shape))
            print('right_embedded shape: {}'.format(right_embedded.shape))

            real_labels = torch.ones(right_images.size(0), device=device)
            fake_lables = torch.zeros(right_images.size(0), device=device)

            # ======== One sided label smoothing ==========
            # Helps preventing the discriminator from overpowering the
            # generator adding penalty when the discriminator is too confident
            # =============================================

            smoothed_real_lables = real_labels - 0.1

            # train the discriminator
            discriminator.zero_grad()
            outputs, activation_real = discriminator(right_images, right_embedded)
            real_loss = criterion(output, smoothed_real_lables)
            real_score = outputs

            outputs, _ = discriminator(wrong_images, right_embedded)
            wrong_loss = criterion(outputs, fake_lables)
            wrong_score = outputs

            noise = torch.randn(right_images(0), 100, device=device)
            noise = noise.view(noise.size(0), 100, 1, 1)
            print('noise shape: {}'.format(noise.shape))

            fake_images = generator(right_embedded, noise)
            outputs, _ = discriminator(fake_images, right_embedded)
            fake_loss = criterion(outputs, fake_lables)
            fake_score = outputs

            d_loss = real_loss + fake_loss + wrong_loss

            d_loss.backwrad()

            optimizerD.step()

            # train the generator

            generator.zero_grad()
            noise = troch.randn(right_images(0), 100, device=device)
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = generator(right_embedded, noise)
            outputs, activation_fake = discriminator(fake_images, right_embedded)
            _, activation_real = discriminator(right_images, right_embedded)

            activation_fake = torch.mean(activation_fake, 0)
            activation_real = torch.mean(activation_real, 0)

            g_loss = criterion(outputs, real_labels) \
                     + l2_coef * l2_criterion(activation_fake, activation_real) \
                     + l1_coef * l1_criterion(fake_images, right_images)

            g_loss.backward()

            optimizerG.step()

            if iteration % 5 == 0:
                self.logger.log_iteration_gan(epoch,d_loss, g_loss, real_score, fake_score)


if __name__ == '__main__':
    train()
