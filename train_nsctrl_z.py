import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torch import nn

import numpy as np

from datasets.simulation import SimulationDataset
from models.metrics.hmm_metrics import compute_acc
from models.utils import JacobianMLP, MLP


class SparseZ(pl.LightningModule):
    def __init__(self, n_class=5, x_dim=7, lags=1, hidden_dim=32, lr=1e-3):
        super().__init__()
        self.lags = lags
        self.n_class = n_class
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.jacobian_mlps = nn.ModuleList(
            [
                JacobianMLP(jacobian_support=np.eye(x_dim), hid_dim=hidden_dim)
                for _ in range(n_class)
            ]
        )
        self.gating = MLP(x_dim * 2, hidden_dim, n_class, depth=1)
        self.criteria = nn.MSELoss()
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        _, x, _ = batch
        batch_size, lags_and_length, _ = x.shape
        length = lags_and_length - self.lags
        x_pair = (
            x.unfold(dimension=1, size=self.lags + 1, step=1)
            .transpose(-2, -1)
            .reshape(batch_size, length, -1)
        )
        C_x_pair = self.gating(x_pair)
        # gumbel_softmax
        C_x_pair = torch.nn.functional.gumbel_softmax(C_x_pair, tau=1, hard=True)
        x_prev = x[:, : -self.lags]
        x_curr = x[:, self.lags :]
        x_hat_curr = torch.zeros_like(x_curr)
        for i in range(self.n_class):
            x_hat_curr += C_x_pair[:, :, i].unsqueeze(-1) * self.jacobian_mlps[i](
                x_prev
            )
        loss = self.criteria(x_hat_curr, x_curr)
        self.log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, c = batch
        batch_size, lags_and_length, _ = x.shape
        length = lags_and_length - self.lags
        x_pair = (
            x.unfold(dimension=1, size=self.lags + 1, step=1)
            .transpose(-2, -1)
            .reshape(batch_size, length, -1)
        )
        C_x_pair = self.gating(x_pair)
        # gumbel_softmax
        C_x_pair = torch.nn.functional.gumbel_softmax(C_x_pair, tau=1, hard=True)
        x_prev = x[:, : -self.lags]
        x_curr = x[:, self.lags :]
        x_hat_curr = torch.zeros_like(x_curr)
        for i in range(self.n_class):
            x_hat_curr += C_x_pair[:, :, i].unsqueeze(-1) * self.jacobian_mlps[i](
                x_prev
            )
        loss = self.criteria(x_hat_curr, x_curr)
        self.log_dict({"val/loss": loss})
        c_est = C_x_pair.argmax(dim=-1)
        self.validation_step_outputs.append(
            {"loss": loss.data, "c": c.cpu().numpy(), "c_est": c_est.cpu().numpy()}
        )

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        c = np.concatenate([x["c"] for x in outputs], axis=0)
        c_est = np.concatenate([x["c_est"] for x in outputs], axis=0)
        # c_est_trans = np.array([indexes[c] for c in c_est.flatten()])
        # acc = np.mean((c_est_trans == c.flatten()).astype(int))
        acc, matchidx = compute_acc(c, c_est, C=self.n_class)
        self.log_dict({"val/acc": acc}, prog_bar=True)
        self.log_dict({"val/loss": avg_loss}, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0001,
        )
        return [opt_v], []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=770)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

n_class = 5
lags = 1
z_dim = 8
lr = 5e-4
hidden_dim = 32
dataset_path = f"data/simulation/C_{n_class}_z_dim_{z_dim}_hid_dim_32_num_batches_1000"

pl.seed_everything(args.seed)
dataset = SimulationDataset(dataset_path)
n_validation = int(0.2 * len(dataset))
train_data, valid_data = random_split(
    dataset, [len(dataset) - n_validation, n_validation]
)

train_loader = DataLoader(
    train_data, shuffle=True, batch_size=64, pin_memory=True, num_workers=4
)
valid_loader = DataLoader(
    valid_data, shuffle=False, batch_size=256, pin_memory=True, num_workers=4
)
model = SparseZ(n_class=n_class, x_dim=z_dim, lags=lags, hidden_dim=hidden_dim, lr=lr)
trainer = pl.Trainer(
    default_root_dir=f"outputs/nsctrl_z/{args.seed}",
    max_epochs=500,
    val_check_interval=0.1,
    accelerator="gpu",
    devices=[args.device],
    deterministic=True
)
trainer.fit(model, train_loader, valid_loader)
