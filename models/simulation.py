import torch
import lightning.pytorch as pl
from torch import nn
import torch.distributions as tD
import numpy as np
from .metrics.correlation import compute_mcc
from .metrics.hmm_metrics import compute_acc
from .utils import JacobianMLP, MLP, BetaVAE_MLP, NPChangeTransitionPrior


class NSCTRL(pl.LightningModule):
    def __init__(
        self,
        n_class=5,
        x_dim=7,
        z_dim=7,
        lags=1,
        hidden_dim=32,
        embedding_dim=2,
        alpha=1.0,
        beta=0.002,
        gamma=0.02,
        lr=1e-3,
        correlation="Pearson",
        weight_decay=1e-4,
    ):
        super().__init__()
        self.lags = lags
        self.n_class = n_class
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.correlation = correlation
        self.weight_decay = weight_decay
        self.jacobian_mlps = nn.ModuleList(
            [
                JacobianMLP(jacobian_support=np.eye(z_dim), hid_dim=hidden_dim)
                for _ in range(n_class)
            ]
        )
        self.gating = MLP(z_dim * 2, hidden_dim, n_class, depth=1)
        self.criteria = nn.MSELoss()
        self.net = BetaVAE_MLP(x_dim, z_dim, hidden_dim, leaky_relu_slope=0.2)
        self.c_embeddings = nn.Embedding(n_class, embedding_dim)
        self.transition_prior = NPChangeTransitionPrior(
            lags=lags,
            latent_size=z_dim,
            embedding_dim=embedding_dim,
            num_layers=2,
            hidden_dim=hidden_dim,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # base distribution for calculation of log prob under the model
        # self.register_buffer("base_dist_mean", torch.zeros(z_dim))
        # self.register_buffer("base_dist_var", torch.eye(z_dim))
        self.best_mcc = -np.inf
        self.validation_step_outputs = []
        # self.automatic_optimization = False

    # @property
    # def base_dist(self):
    #     return tD.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    # def tdrl_loss(self, mus, logvars, z_est, c_est):
    #     _, lags_and_length, _ = mus.shape

    #     embeddings = self.c_embeddings(c_est)

    #     q_dist = tD.Normal(mus, torch.exp(logvars / 2))
    #     log_qz = q_dist.log_prob(z_est)

    #     # Past KLD
    #     p_dist = tD.Normal(
    #         torch.zeros_like(mus[:, : self.lags]),
    #         torch.ones_like(logvars[:, : self.lags]),
    #     )
    #     log_pz_normal = torch.sum(
    #         torch.sum(p_dist.log_prob(z_est[:, : self.lags]), dim=-1), dim=-1
    #     )
    #     log_qz_normal = torch.sum(torch.sum(log_qz[:, : self.lags], dim=-1), dim=-1)
    #     kld_normal = log_qz_normal - log_pz_normal
    #     kld_normal = kld_normal.mean()
    #     # Future KLD
    #     log_qz_laplace = log_qz[:, self.lags :]

    #     residuals, logabsdet = self.transition_prior(z_est, embeddings)

    #     log_pz_laplace = torch.sum(
    #         self.base_dist.log_prob(residuals), dim=1
    #     ) + logabsdet.sum(dim=1)
    #     kld_laplace = (
    #         torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace
    #     ) / (lags_and_length - self.lags)
    #     kld_laplace = kld_laplace.mean()

    #     return kld_normal, kld_laplace

    def training_step(self, batch, batch_idx):

        x, z, c = batch
        batch_size, lags_and_length, _ = x.shape
        length = lags_and_length - self.lags

        x_recon, mus, logvars, z_est = self.net(x)
        recon_loss = self.criteria(x_recon, x)

        z_hat = mus
        z_est_pair = (
            z_hat.unfold(dimension=1, size=self.lags + 1, step=1)
            .transpose(-2, -1)
            .reshape(batch_size, length, -1)
        )
        C_z_pair = self.gating(z_est_pair)
        # gumbel_softmax
        C_z_pair = torch.nn.functional.gumbel_softmax(C_z_pair, tau=1, hard=True)
        z_prev = z_hat[:, : -self.lags]
        z_curr = z_hat[:, self.lags :]
        z_hat_curr = torch.zeros_like(z_curr)

        for i in range(self.n_class):
            z_hat_curr += C_z_pair[:, :, i].unsqueeze(-1) * self.jacobian_mlps[i](
                z_prev
            )
        transition_loss = self.criteria(z_hat_curr, z_curr)
        sparsity_loss = sum([mlp.get_l1() for mlp in self.jacobian_mlps])
        # c_est = C_z_pair.argmax(dim=-1)

        # past_kld, future_kld = self.tdrl_loss(mus, logvars, z_est, c_est)

        loss = (
            recon_loss
            + self.alpha * transition_loss
            + sparsity_loss
            # + self.beta * past_kld
            # + self.gamma * future_kld
        )

        self.log_dict(
            {
                "train/loss": loss,
                "train/recon_loss": recon_loss,
                "train/transition_loss": transition_loss,
                # "train/past_kld": past_kld,
                # "train/future_kld": future_kld,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, z, c = batch
        batch_size, lags_and_length, _ = x.shape
        length = lags_and_length - self.lags

        x_recon, mus, logvars, z_est = self.net(x)
        recon_loss = self.criteria(x_recon, x)
        z_hat = mus
        z_est_pair = (
            z_hat.unfold(dimension=1, size=self.lags + 1, step=1)
            .transpose(-2, -1)
            .reshape(batch_size, length, -1)
        )
        C_z_pair = self.gating(z_est_pair)
        # gumbel_softmax
        C_z_pair = torch.nn.functional.gumbel_softmax(C_z_pair, tau=1, hard=True)
        z_prev = z_hat[:, : -self.lags]
        z_curr = z_hat[:, self.lags :]
        z_hat_curr = torch.zeros_like(z_curr)
        for i in range(self.n_class):
            z_hat_curr += C_z_pair[:, :, i].unsqueeze(-1) * self.jacobian_mlps[i](
                z_prev
            )
        transition_loss = self.criteria(z_hat_curr, z_curr)
        sparsity_loss = sum([mlp.get_l1() for mlp in self.jacobian_mlps])
        c_est = C_z_pair.argmax(dim=-1)
        # past_kld, future_kld = self.tdrl_loss(mus, logvars, z_est, c_est)

        loss = (
            recon_loss
            + self.alpha * transition_loss
            + sparsity_loss
            # + self.beta * past_kld
            # + self.gamma * future_kld
        )

        self.validation_step_outputs.append(
            {
                "loss": loss.data,
                "recon_loss": recon_loss.data,
                "transition_loss": transition_loss.data,
                "sparsity_loss": sparsity_loss.data,
                # "past_kld": past_kld.data,
                # "future_kld": future_kld.data,
                "c": c.cpu().numpy(),
                "c_est": c_est.cpu().numpy(),
                "z": z.cpu().numpy(),
                "z_est": mus.cpu().numpy(),
            }
        )

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        # avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x["recon_loss"] for x in outputs]).mean()
        avg_transition_loss = torch.stack(
            [x["transition_loss"] for x in outputs]
        ).mean()
        avg_sparsity_loss = torch.stack([x["sparsity_loss"] for x in outputs]).mean()
        # avg_past_kld = torch.stack([x["past_kld"] for x in outputs]).mean()
        # avg_future_kld = torch.stack([x["future_kld"] for x in outputs]).mean()

        self.log_dict(
            {
                "r_l": avg_recon_loss,
                "t_l": avg_transition_loss,
                "s_l": avg_sparsity_loss,
                # "p_l": avg_past_kld,
                # "f_l": avg_future_kld,
            },
            prog_bar=True,
            sync_dist=True,
        )
        c = np.concatenate([x["c"] for x in outputs], axis=0)
        c_est = np.concatenate([x["c_est"] for x in outputs], axis=0)

        z = np.concatenate([x["z"] for x in outputs], axis=0)

        z_est = np.concatenate([x["z_est"] for x in outputs], axis=0)
        z = z[:, self.lags :].reshape(-1, z.shape[-1]).T
        z_est = z_est[:, self.lags :].reshape(-1, z_est.shape[-1]).T
        acc, matchidx = compute_acc(c, c_est, C=self.n_class)
        mcc = compute_mcc(z_est, z, self.correlation)
        if mcc > self.best_mcc:
            self.best_mcc = mcc

        self.log_dict(
            {"acc": acc, "mcc": mcc, "best_mcc": self.best_mcc},
            prog_bar=True,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()

    def configure_optimizers(self):

        opt_v = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        return [opt_v], []
