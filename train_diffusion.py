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
import matplotlib.pyplot as plt


def train():

    device = 'cuda'
    pizza_data_dir = './data/pizza_data/images'
    dataset = PizzaDataset(pizza_data_dir)

    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    config_path = './config/unet-config.yaml'
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    unet = UNetModel(config).to(device)
    sampler = DiffusionSampler(unet).to(device)
    sampler.sampling_steps = 50

    optimizer = torch.optim.Adam(sampler.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    chkpt_path = './data/checkpoints/sampler-latest.pt'

    epoch_start = 1
    for epoch in range(epoch_start, epoch_start + 100):

        sampler.train()

        batch_losses = []
        train_progress_bar = tqdm(dataloader)

        for i, batch in enumerate(train_progress_bar):
            try:
                optimizer.zero_grad()
                batch = batch.to(device)
                batch = batch * 2.0 - 1.0
                t = torch.randint(0, sampler.timesteps, (batch_size,), device=device).long()
                loss = sampler.p_loss(batch, t)
                batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()

                train_progress_bar.set_description(f'Epoch {epoch}, batch_loss: {loss.item():.4f}')
            except:
                continue

        torch.save(sampler.state_dict(), chkpt_path)

        sampler.eval()
        with torch.no_grad():
            shape = (1, 3, 512, 512)
            for j in range(10):
                output = sampler.ddim_sample(shape)
                output = (output + 1.0) / 2.0
                output = output[0]
                output = output.permute(1, 2, 0).cpu().numpy()
                plt.imsave(f'./data/samples/ep{epoch}-{j}.png', output)





if __name__ == "__main__":
    train()

