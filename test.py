import os
import numpy as np
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import Unet
from utils.helpers import sample_timestep
from utils.plots import plot_error

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_transform = transforms.Compose([transforms.ToTensor()])

def load_model(path, device):
    model = Unet(dim=32).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def main():
    model_path = '/path/to/UNET_BASED_step=1_code.pth'
    root_dir = '/path/to/test_image/compounded_test_image2_unnormalized'
    model = load_model(model_path, device)
    model.eval()

    angle = 3
    sigma_ratio = 1
    error_list = []

    image1 = load_image(root_dir, '1', 'image_0002.mat')
    image2 = load_image(root_dir, '15', 'image_0002.mat')

    image1, image2 = data_transform(image1), data_transform(image2)
    image1 = image1.reshape(1, *image1.shape).float().to(device)
    image2 = image2.reshape(1, *image2.shape).float().to(device)

    input_image = image1.clone()
    diff = (image1 - image2) / sigma_ratio

    for i in range(sigma_ratio, 0, -1):
        t = torch.tensor([i], device=device)
        output = sample_timestep(model, input_image, t)
        input_image -= output
        error = torch.nn.functional.l1_loss(input_image[0][0].cpu(), image2[0][0].cpu())
        error_list.append(error.item())

        print(f"Iteration {sigma_ratio - i + 1} | Error: {error.item()}")

    save_results(image1, image2, input_image)
    plot_error(error_list, ylabel="Distance with 75 angles in each iteration of inverse process")

def load_image(root_dir, subdir, filename):
    path = os.path.join(root_dir, subdir, filename)
    return loadmat(path)['DASRFs'][:1072, :]

def save_results(image1, image2, output):
    results = {
        'image1': image1[0][0].cpu().numpy(),
        'image2': image2[0][0].cpu().numpy(),
        'output': output[0][0].cpu().numpy()
    }
    savemat("/path/to/results/UNET_robust_based_normalized.mat", results)

if __name__ == "__main__":
    main()
