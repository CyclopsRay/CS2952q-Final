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


def sample():
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

    optimizer = torch.optim.Adam(sampler.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    chkpt_path = './data/checkpoints/sampler-latest.pt'
    sampler.load_state_dict(torch.load(chkpt_path))

    sampler.sampling_steps = 50

    shape = (1, 3, 256, 256)

    for i in range(100):
        output = sampler.ddim_sample(shape)
        output = (output + 1.0) / 2.0
        output = output[0]
        output = output.permute(1, 2, 0).cpu().numpy()
        plt.imsave(f'./data/sample256/{i}.png', output)


@torch.no_grad()
def add_noise_and_denoise(sampler, img, t_level):
    sampler.sampling_steps = 1000

    device = img.device

    # normalize image
    img = img * 2.0 - 1.0

    # shape (1, 3, H, W)
    img = img.unsqueeze(0)

    # cast int t to batched tensor shape: (1,)
    t_tensor = torch.full((1,), t_level, device=device, dtype=torch.long)

    # initialize noise
    noise = torch.randn_like(img).to(device)

    # add noise
    img_noisy = sampler.q_sample(img, t_tensor, noise)

    # denoise
    output = img_noisy.clone()
    img_noisy = (img_noisy + 1.0) / 2.0

    times = torch.linspace(-1, t_level - 1, t_level + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))

    sampling_progress_bar = tqdm(time_pairs, desc='Denoising')

    for i, (t, t_m1) in enumerate(sampling_progress_bar):

        time_cond = torch.full((1,), t, device=device, dtype=torch.long)

        # x_start is the less noisy image after removing pred_noise
        # pred_noise is the output of the model
        pred_noise, x_start = sampler.model_predictions(output, time_cond)

        if t_m1 < 0:
            output = x_start
            continue

        alpha_bar = sampler.alphas_cumprod[t]
        alpha_next_bar = sampler.alphas_cumprod[t_m1]

        sigma = torch.sqrt(
            sampler.ddim_eta * ((1 - alpha_bar / alpha_next_bar) * (1 - alpha_next_bar) / (1 - alpha_bar)))
        c = torch.sqrt(1 - alpha_next_bar - sigma ** 2)

        noise = torch.randn_like(output)

        output = x_start * torch.sqrt(alpha_next_bar) + \
                 c * pred_noise + \
                 sigma * noise

    # cast back to [0, 1] range
    output = (output + 1.0) / 2.0

    img_noisy = torch.clamp(img_noisy, 0.0, 1.0)
    output = torch.clamp(output, 0.0, 1.0)

    # print('noisy', img_noisy.min(), img_noisy.max())
    # print('outpput', output.min(), output.max())

    return output, img_noisy


if __name__ == '__main__':
    # sample()

    device = 'cuda'

    config_path = './config/unet-config.yaml'
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    unet = UNetModel(config).to(device)
    sampler = DiffusionSampler(unet).to(device)

    chkpt_path = './data/checkpoints/sampler-latest.pt'
    sampler.load_state_dict(torch.load(chkpt_path))

    img_path = './data/pizza_data/images/00004.jpg'
    img = transforms.ToTensor()(PIL.Image.open(img_path))
    img = transforms.Resize((256, 256))(img)
    img = img[:3].to(device)

    t_level = 50

    output, img_noisy = add_noise_and_denoise(sampler, img, t_level)

    img = img.permute(1, 2, 0).cpu().numpy()
    output = output[0].permute(1, 2, 0).cpu().numpy()
    img_noisy = img_noisy[0].permute(1, 2, 0).cpu().numpy()

    img = np.concatenate([img, img_noisy, output], axis=1)

    plt.imsave('test.png', img)


