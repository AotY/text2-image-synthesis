#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,
                 image_size,
                 num_channels,
                 noise_size,
                 embedding_size,
                 projected_embedding_size,
                 ngf):
        super(Generator, self).__init__()

        self.image_size = image_size
        self.num_channels = num_channels
        self.noise_size = noise_size
        self.embedding_size = embedding_size
        self.projected_embedding_size = projected_embedding_size
        self.lantent_size = noise_size + projected_embedding_size
        self.ngf = ngf

        # embedding_size -> projected_embedding_size
        self.projector = nn.Sequential(
            nn.Linear(embedding_size, projected_embedding_size),
            nn.BatchNorm1d(projected_embedding_size),
            nn.LeakyReLU(0.2, True)
        )

        self.netG = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.lantent_size,
                out_channels=ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False), # [ngf*8, h_, w, ]
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True), # inplace,
            # state size [ngf*8, 4, 4]

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size [ngf * 4, 8, 8]

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size [ngf * 2, 16, 16]

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size [ngf, 32, 32]

            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size [num_channels, 64, 64]
        )


    def forward(self, embedded, noise):
        projected_embedded = self.projector(embedded) # []
        projected_embedded.unsqueeze_(2) # [, , 1]
        projected_embedded.unsqueeze_(3) # [, , 1, 1]

        lantent_embedded = torch.cat((projected_embedded, noise), dim=1)

        output = self.netG(lantent_embedded)

        return output


class Discriminator(nn.Module):
    def __init__(self,
                 image_size,
                 num_channels,
                 embedding_size,
                 projected_embedding_size,
                 ndf,
                 b_size,
                 c_size):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.projected_embedding_size = projected_embedding_size
        self.ndf = ndf
        self.b_size = b_size
        self.c_size = c_size


        self.netD_1 = nn.Sequential (
            # input: [num_channels, 64, 64]
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # state size: [ndf, 32, 32]

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # state size: [ndf * 2, 16, 16]

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # state size: [ndf * 4, 8, 8]

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
            # state size: [ndf * 8, 4, 4]
        )

        self.projector = nn.Sequential(
            nn.Linear(embedding_size, projected_embedding_size),
            nn.BatchNorm1d(projected_embedding_size),
            nn.LeakyReLU(0.2, True)
        )

        self.netD_2 = nn.Sequential(
            # state size: [ndf * 8, 4, 4]
            nn.Conv2d(ndf * 8 + projected_embedding_size, 1, 4, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input, embedded):
        # input: [num_channels, 64, 64]
        output_tmp = self.netD_1(input)

        projected_embedded = self.projector(embedded)
        replicated_embedded = projected_embedded.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        hidden_concat = torch.cat((input, replicated_embedded), dim=1)

        output = self.netD_2(hidden_concat)

        return output.view(-1, 1).squeeze(1), output_tmp
