import torch
import numpy as np
import torch.nn as nn
import torch.distributions as tD


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth=1):
        super(MLP, self).__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            h = self.relu(hidden_layer(h))
        out = self.fc2(h)
        return out


class HMM(nn.Module):
    def __init__(self, n_class, x_dim, lags=1, hidden_dim=32) -> None:
        super().__init__()
        self.log_A = nn.Parameter(torch.randn(n_class, n_class))
        self.log_pi = nn.Parameter(torch.randn(n_class))
        self.n_class = n_class
        self.x_dim = x_dim
        self.lags = lags
        input_size = (self.lags) * x_dim
        output_size = n_class * 2 * x_dim
        self.trans = MLP(input_size, hidden_dim, output_size, depth=2)

    def forward_log(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        # length = lags_and_length - self.lags
        log_alpha = torch.zeros(
            batch_size, length, self.n_class, device=logp_x_c.device
        )
        log_A = torch.log_softmax(self.log_A, dim=1)
        log_pi = torch.log_softmax(self.log_pi, dim=0)
        for t in range(length):
            if t == 0:
                log_alpha_t = logp_x_c[:, t] + log_pi
            else:
                log_alpha_t = logp_x_c[:, t] + torch.logsumexp(
                    log_alpha[:, t - 1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1
                )
            log_alpha[:, t] = log_alpha_t
        logp_x = torch.logsumexp(log_alpha[:, -1], dim=-1)
        # logp_x = torch.sum(log_scalers, dim=-1)
        return logp_x

    def forward_backward_log(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        # length = lags_and_length - self.lags
        log_alpha = torch.zeros(
            batch_size, length, self.n_class, device=logp_x_c.device
        )
        log_beta = torch.zeros(batch_size, length, self.n_class, device=logp_x_c.device)
        log_scalers = torch.zeros(batch_size, length, device=logp_x_c.device)
        log_A = torch.log_softmax(self.log_A, dim=1)
        log_pi = torch.log_softmax(self.log_pi, dim=0)
        for t in range(length):
            if t == 0:
                log_alpha_t = logp_x_c[:, t] + log_pi
            else:
                log_alpha_t = logp_x_c[:, t] + torch.logsumexp(
                    log_alpha[:, t - 1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1
                )
            log_scalers[:, t] = torch.logsumexp(log_alpha_t, dim=-1)
            log_alpha[:, t] = log_alpha_t - log_scalers[:, t].unsqueeze(-1)
        log_beta[:, -1] = torch.zeros(batch_size, self.n_class, device=logp_x_c.device)
        for t in range(length - 2, -1, -1):
            log_beta_t = torch.logsumexp(
                log_beta[:, t + 1].unsqueeze(-1)
                + log_A.unsqueeze(0)
                + logp_x_c[:, t + 1].unsqueeze(1),
                dim=-1,
            )
            log_beta[:, t] = log_beta_t - log_scalers[:, t].unsqueeze(-1)
        log_gamma = log_alpha + log_beta
        # logp_x = torch.logsumexp(log_alpha[:, -1],dim=-1)
        logp_x = torch.sum(log_scalers, dim=-1)
        return log_alpha, log_beta, log_scalers, log_gamma, logp_x

    def viterbi_algm(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        log_delta = torch.zeros(
            batch_size, length, self.n_class, device=logp_x_c.device
        )
        psi = torch.zeros(
            batch_size, length, self.n_class, dtype=torch.long, device=logp_x_c.device
        )

        log_A = torch.log_softmax(self.log_A, dim=1)
        log_pi = torch.log_softmax(self.log_pi, dim=0)
        for t in range(length):
            if t == 0:
                log_delta[:, t] = logp_x_c[:, t] + log_pi
            else:
                max_val, max_arg = torch.max(
                    log_delta[:, t - 1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1
                )
                log_delta[:, t] = max_val + logp_x_c[:, t]
                psi[:, t] = max_arg
        c = torch.zeros(batch_size, length, dtype=torch.long, device=logp_x_c.device)
        c[:, -1] = torch.argmax(log_delta[:, -1], dim=-1)
        for t in range(length - 2, -1, -1):
            c[:, t] = psi[:, t + 1].gather(1, c[:, t + 1].unsqueeze(1)).squeeze()
        return c  # , logp_x

    def forward(self, x):
        batch_size, lags_and_length, _ = x.shape
        length = lags_and_length - self.lags
        # x_H = (batch_size, length, (lags) * x_dim)
        x_H = x.unfold(dimension=1, size=self.lags + 1, step=1).transpose(-2, -1)

        x_H = x_H[..., : self.lags, :].reshape(batch_size, length, -1)
        # x_H = x_H.reshape(batch_size, length, -1)

        # (batch_size, length, n_class, x_dim)
        out = self.trans(x_H).reshape(batch_size, length, self.n_class, 2 * self.x_dim)
        mus, logvars = out[..., : self.x_dim], out[..., self.x_dim :]
        dist = tD.Normal(mus, torch.exp(logvars / 2))
        logp_x_c = dist.log_prob(x[:, self.lags :].unsqueeze(2)).sum(
            -1
        )  # (batch_size, length, n_class)
        log_alpha, log_beta, log_scalers, log_gamma, logp_x = self.forward_backward_log(
            logp_x_c
        )
        # elif self.mode == "mle":
        # logp_x = self.forward_log(logp_x_c)

        c_est = self.viterbi_algm(logp_x_c)
        return logp_x, c_est
