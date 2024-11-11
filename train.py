import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from data.dataset import ANGLES
from models.unet import Unet
from utils.helpers import get_loss
from utils.plots import plot_loss

BATCH_SIZE = 4
EPOCHS = 500
SIGMA_R = 1


def train_val_split(dataset, val_split=0.01):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = '/path/to/data'
    data = ANGLES(root_dir)
    train_data, val_data = train_val_split(data)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = Unet(dim=32).to(device)
    optimizer = Adam(model.parameters(), lr=0.004)
    train_list, test_list = [], []

    for epoch in range(EPOCHS):
        train_loss = 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            t = torch.randint(1, SIGMA_R + 1, (BATCH_SIZE,)).to(device).view(BATCH_SIZE, 1, 1, 1)
            loss = get_loss(model, batch[1].to(device), batch[3].to(device), t, sigma_ratio=1, device=device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_list.append(train_loss)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                t = torch.randint(1, SIGMA_R + 1, (BATCH_SIZE,)).to(device).view(BATCH_SIZE, 1, 1, 1)
                loss = get_loss(model, batch[1].to(device), batch[3].to(device), t, sigma_ratio=1, device=device)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        test_list.append(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss}")

    plot_loss(train_list, test_list)


if __name__ == "__main__":
    main()
