import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def get_beta_schedule(num_steps, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start, beta_end, num_steps)


def forward_diffusion(img, betas):
    xs = [img]
    noises = []
    sqrt_alphas = np.sqrt(1 - betas)
    sqrt_one_minus_alphas = np.sqrt(betas)
    x = img.astype(np.float32) / 255.0
    for t in range(len(betas)):
        noise = np.random.randn(*x.shape)
        x = sqrt_alphas[t] * x + sqrt_one_minus_alphas[t] * noise
        xs.append(np.clip(x, 0.0, 1.0))
        noises.append(noise)
    return xs, noises


def reverse_diffusion(xs, noises, betas):
    xr = [xs[-1]]
    sqrt_alphas = np.sqrt(1 - betas)
    sqrt_one_minus_alphas = np.sqrt(betas)
    x = xs[-1]
    for t in reversed(range(len(betas))):
        noise = noises[t]
        x = (x - sqrt_one_minus_alphas[t] * noise) / sqrt_alphas[t]
        xr.append(np.clip(x, 0.0, 1.0))
    return xr[::-1]


def plot_process(forward_steps, reverse_steps, save_path):
    num_steps = len(forward_steps)
    fig, axes = plt.subplots(2, num_steps, figsize=(2 * num_steps, 4))
    for i in range(num_steps):
        axes[0, i].imshow(forward_steps[i], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'$x_{i}$')

        axes[1, i].imshow(reverse_steps[i], cmap='gray')
        axes[1, i].axis('off')
    axes[1,0].set_ylabel('Reverse', fontsize=12)
    axes[0,0].set_ylabel('Forward', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")



def main():
    img_path = os.path.join("data", "toy", "img_checkerboard.png")
    img = np.array(Image.open(img_path).convert("L"))
    num_steps = 6
    betas = get_beta_schedule(num_steps)
    forward_steps, noises = forward_diffusion(img, betas)
    reverse_steps = reverse_diffusion(forward_steps, noises, betas)
    plot_process(forward_steps, reverse_steps, os.path.join("img", "forward_backward.png"))


if __name__ == "__main__":
    main()
