import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from src.sae_model import SparseAutoencoder
from src.sbert_encoder import SBERTEncoder

class PromptDataset(Dataset):
    def __init__(self, pkl_path):
        df = pd.read_pickle(pkl_path)
        self.prompts = df["prompt"].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def train_sae():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[SAE] Loading SBERT...")
    sbert = SBERTEncoder(device=device)

    print("[SAE] Loading RouterBench prompts...")
    ds = PromptDataset("data/routerbench_0shot.pkl")
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    sae = SparseAutoencoder(input_dim=384, hidden_dim=4096).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    mse = torch.nn.MSELoss()

    λ = 1e-3  # sparsity regularization

    for epoch in range(5):
        total_loss = 0
        total_l1 = 0

        for batch_text in tqdm(dl):
            x = sbert.encode(batch_text).to(device)  # (B, 384)
            x = x.clone().detach()
            h, x_hat = sae(x)

            recon_loss = mse(x_hat, x)
            l1 = h.abs().mean()

            loss = recon_loss + λ * l1

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += recon_loss.item()
            total_l1 += l1.item()

        print(f"[Epoch {epoch+1}] recon={total_loss:.4f}, L1={total_l1:.4f}")

    torch.save(sae.state_dict(), "sae_model.pt")
    print("[SAE] Saved to sae_model.pt")


if __name__ == "__main__":
    train_sae()
