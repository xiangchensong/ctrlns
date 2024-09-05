import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split


from models.hmm import HMM
from models.metrics.hmm_metrics import compute_acc,compute_min_A_err
import numpy as np

from datasets.simulation import SimulationDataset

torch.use_deterministic_algorithms(True)

class HMMz(pl.LightningModule):
    def __init__(
            self,
            n_class=5,
            x_dim=7,
            lags=1,
            hidden_dim=32,
            lr=1e-3):
        super().__init__()
        self.A = None
        self.lags = lags
        self.n_class = n_class
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hmm = HMM(n_class=self.n_class, x_dim=x_dim, lags=lags, hidden_dim=hidden_dim)
        
        self.validation_step_outputs = []
        

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        # _, lags_and_length, _ = x.shape
        E_logp_x, c_est = self.hmm(x)
        hmm_loss = - E_logp_x.mean()
        self.log_dict({'train/loss': hmm_loss})
        return hmm_loss

    def validation_step(self, batch, batch_idx):
        x, z, c = batch
        # _, lags_and_length, _ = x.shape
        E_logp_x, c_est = self.hmm(x)
        hmm_loss = -E_logp_x.mean()
        self.validation_step_outputs.append({'loss': hmm_loss.data, 'c':c.cpu().numpy(), 'c_est':c_est.cpu().numpy()})

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        A_est = torch.log_softmax(self.hmm.log_A.detach(),dim=1).exp().cpu().numpy()
        A = self.A

        c = np.concatenate([x['c'] for x in outputs],axis=0)
        c_est = np.concatenate([x['c_est'] for x in outputs],axis=0)
        # c_est_trans = np.array([indexes[c] for c in c_est.flatten()])
        # acc = np.mean((c_est_trans == c.flatten()).astype(int))
        acc, matchidx = compute_acc(c, c_est, C=self.n_class)
        if acc == 1.0:
           aaa = 1
        A_permuted = A[matchidx,:][:,matchidx]
        A_err = np.abs(A_permuted - A_est).mean()
        print(A_permuted.round(2))
        print(A_est.round(2))
        self.log_dict({'val/acc': acc,
                       'val/A_err': A_err}, prog_bar=True)
        self.log_dict({'val/loss': avg_loss}, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters(
        )), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
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
hidden_dim = 128
dataset_path = f"data/simulation/C_{n_class}_z_dim_{z_dim}_hid_dim_32_num_batches_1000"

pl.seed_everything(args.seed)
dataset = SimulationDataset(dataset_path)
n_validation = int(0.1*len(dataset))
train_data, valid_data = random_split(
    dataset, [len(dataset) - n_validation, n_validation])

train_loader = DataLoader(train_data,
                            shuffle=True,
                            batch_size=64,
                            pin_memory=True)
valid_loader = DataLoader(valid_data,
                            shuffle=False,
                            batch_size=256,
                            pin_memory=True)
model = HMMz(n_class=n_class, x_dim=z_dim, lags=lags, hidden_dim=hidden_dim, lr=lr)
A = np.random.randn(*dataset.transition_matrix.shape)
model.A= A / A.sum(axis=1, keepdims=True)
trainer = pl.Trainer(default_root_dir=f'outputs/hmm/{args.seed}',
                     max_epochs=1000,
                     val_check_interval=0.1,
                     accelerator='gpu',
                     devices=[args.device],
                     deterministic=True)
trainer.fit(model, train_loader, valid_loader)