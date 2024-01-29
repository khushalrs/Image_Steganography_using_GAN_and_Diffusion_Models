import torch.nn.functional as F
import torch
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from PIL import Image
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt

def linear_beta_schedule(start, end, timesteps):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def load_transformed_dataset(img_size=128, device="cuda",max_images = 1000):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
    data_transform = transforms.Compose(data_transforms)

    dataset_folder = "/content/img_align_celeba/img_align_celeba"
    dataset = []
    i = 0
    for filename in os.listdir(dataset_folder):
      try:
          if i == max_images:
              break
          i += 1
          image = Image.open(os.path.join(dataset_folder, filename))
          image = data_transform(image)
          dataset.append((image.to(device), 0))  # Assuming label 0
      except Exception as e:
          print(f"Skipped an image due to an exception: {str(e)}")
      #print(f"Loaded {i} images")

    #print(len(dataset))
    return dataset

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take the first image of the batch
    if len(image.shape) == 4:
      image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(
        device) * noise.to(device), noise

def forward_diffusion_process(image, timesteps, img_size=256):
    for t in range(timesteps):
        image, _ = forward_diffusion_sample(image, torch.Tensor([t]).type(torch.int64))

    final_image = transforms.functional.to_pil_image(image)
    final_image = transforms.functional.resize(final_image, (img_size, img_size))
    return final_image

def save_forward_diffusion_images(output_directory, dataloader, timesteps):
    os.makedirs(output_directory, exist_ok=True)
    j = 0
    for batch in dataloader:
        images, _ = batch
        for image in images:
            final_image = forward_diffusion_process(image, timesteps)
            save_path = os.path.join(output_directory, f"image_{j:06d}.png")
            final_image.save(save_path)
            j += 1

device = "cuda"
img_size = 32
batch_size = 16
timesteps = 100
betas = linear_beta_schedule(0.0001, 0.02, timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
dataloader = DataLoader(load_transformed_dataset(img_size=img_size, device=device), batch_size=batch_size, shuffle=True, drop_last=True)
output_directory = '/content/drive/MyDrive/Colab Notebooks/Capstone/FD images'
plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(timesteps/num_images)
image = next(iter(dataloader))[0]

for idx in range(0, timesteps, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_diffusion_sample(image, t)
    show_tensor_image(img)