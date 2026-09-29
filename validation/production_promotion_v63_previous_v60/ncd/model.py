"""Permutation-invariant DeepSets with exact variable-swap equivariance."""
from pathlib import Path
import copy
import random
import numpy as np
import torch
from torch import nn
from .io import save_json

SWAP = [1, 0, 2, 3]

class Discoverer(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.width = width
        self.encoder = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(2 * width + 2, width), nn.Tanh(), nn.Linear(width, 4))

    def representation(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        z = ((x - mean) / std).clamp(-20, 20)
        h = self.encoder(z)
        # Explicitly retain scales; the audit can detect their use as shortcuts.
        return torch.cat([h.mean(1), h.square().mean(1), std[:, 0].log()], dim=1)

    def hidden_pair(self, x):
        return self.representation(x), self.representation(x.flip(-1))

    def from_hidden(self, h, hs):
        return .5 * (self.head(h) + self.head(hs)[:, SWAP])

    def forward(self, x):
        return self.from_hidden(*self.hidden_pair(x))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

def predict(model, data, batch_size=128):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.as_tensor(data[i:i+batch_size], dtype=torch.float32)).softmax(-1)
            for i in range(0, len(data), batch_size)]).numpy()

def train(data, labels, dev_data, dev_labels, directory, *, epochs=35, width=48, seed=42, batch_size=64):
    if epochs < 1: raise ValueError("epochs must be positive")
    set_seed(seed)
    model = Discoverer(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_t, minlength=4).float().clamp_min(1)
    weights = (len(labels) / (4 * counts))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    x = torch.as_tensor(data, dtype=torch.float32)
    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        losses = []
        for ids in order.split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(x[ids]), labels_t[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        p = predict(model, dev_data)
        dev_loss = float(-np.log(p[np.arange(len(p)), dev_labels].clip(1e-9)).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "dev_loss": dev_loss, "dev_accuracy": float((p.argmax(1) == dev_labels).mean())})
        if dev_loss < best:
            best, best_epoch, best_state = dev_loss, epoch + 1, copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{epochs}: dev_loss={dev_loss:.4f}", flush=True)
    model.load_state_dict(best_state)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"width": width, "state_dict": best_state}, directory / "discoverer.pt")
    save_json(directory / "training.json", {"seed": seed, "best_epoch": best_epoch, "history": history,
        "selection": "minimum unweighted dev cross-entropy", "frozen_after_training": True})
    model.eval()
    return model

def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = Discoverer(state["width"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model
