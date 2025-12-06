import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_latent(h):
    h = h.detach().cpu().numpy()
    idx = np.argsort(h)[::-1][:20]
    vals = h[idx]

    plt.figure(figsize=(8,4))
    plt.title("Top-20 SAE latent activations")
    plt.bar(range(20), vals)
    plt.show()

