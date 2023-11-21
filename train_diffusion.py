import torch
import numpy as np
from torchvision import transforms
import PIL
from torch.utils.data import DataLoader
import os
import sys
import argparse
import random
import yaml
import json
from easydict import EasyDict
from tqdm import tqdm
from models.unetmodules.unet import UNetModel
from models.sampler import DiffusionSampler
from dataset import PizzaDataset


def train():

    device = 'cuda'
    pizza_data_dir = './'
    dataset = PizzaDataset(pizza_data_dir)

    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    config_path = './config/unet-config.yaml'
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    unet = UNetModel(config).to(device)
    sampler = DiffusionSampler(config).to(device)
    sampler.sampling_steps = 100

    optimizer = torch.optim.Adam(sampler.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    chkpt_path = './data/checkpoints/unet-latest.pt'

    epoch_start = 1
    for epoch in range(epoch_start, epoch_start + 100):

        sampler.train()

        batch_losses = []
        train_progress_bar = tqdm(dataloader)

        for i, batch in enumerate(train_progress_bar):
            optimizer.zero_grad()

            batch = batch.to(device)
            t = torch.randint(0, sampler.timesteps, (batch_size,), device=device).long()
            loss = sampler.p_loss(batch, t)
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            train_progress_bar.set_description(f'Epoch {epoch}, batch_loss: {loss.item():.4f}')

        torch.save(sampler.state_dict(), chkpt_path)





if __name__ == "__main__":
    train()

