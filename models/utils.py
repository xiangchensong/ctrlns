import torch
from torch import nn
import torch.nn.init as init
from torch.func import jacfwd, jacrev, vmap


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std * eps


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


class JacobianMLP(nn.Module):
    def __init__(self, jacobian_support, hid_dim):
        super(JacobianMLP, self).__init__()
        jacobian_support = torch.tensor(jacobian_support)
        out_dim, in_dim = jacobian_support.shape
        self.out_dim = out_dim
        self.input_layers = nn.ModuleList()
        self.output_layer = nn.ModuleList()
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.jacobian_support = jacobian_support
        for i in range(out_dim):
            linear_layer = nn.Linear(in_dim, hid_dim)
            # zero out the weights
            linear_layer.weight.data.zero_()
            # Use boolean indexing for efficient weight initialization
            mask = jacobian_support[i] == 1
            assert mask.sum() > 0, "Each output must depend on at least one input"
            normal_weights = torch.randn(hid_dim, in_dim)
            small_values_mask = normal_weights.abs() < 0.01
            adjusted_values = 0.02 * (normal_weights >= 0).float() - 0.01
            normal_weights[small_values_mask] = adjusted_values[small_values_mask]
            normal_weights = normal_weights * mask.float()
            linear_layer.weight.data = normal_weights  # Transpose to match shape

            self.input_layers.append(linear_layer)
            self.output_layer.append(nn.Linear(hid_dim, 1))

    def get_l1(self):
        return sum(
            [
                torch.abs(param[1]).sum()
                for param in self.named_parameters()
                if "weight" in param[0] and "input_layers" in param[0]
            ]
        )

    def forward(self, x):
        outs = []
        for i in range(self.out_dim):
            hidden = self.relu(self.input_layers[i](x))
            out = self.output_layer[i](hidden)
            outs.append(out)
        outs = torch.cat(outs, dim=-1)
        return outs


class BetaVAE_MLP(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=3, z_dim=10, hidden_dim=128, leaky_relu_slope=0.2):
        super(BetaVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, 2 * z_dim),
        )
        # Fix the functional form to ground-truth mixing function
        self.decoder = nn.Sequential(
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, input_dim),
        )
        # self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):

        distributions = self._encode(x)
        mu = distributions[..., : self.z_dim]
        logvar = distributions[..., self.z_dim :]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class NPChangeTransitionPrior(nn.Module):

    def __init__(self, lags, latent_size, embedding_dim, num_layers=3, hidden_dim=64):
        super().__init__()
        self.latent_size = latent_size
        self.lags = lags
        self.gs = nn.ModuleList(
            [
                MLP(hidden_dim + lags * latent_size + 1, hidden_dim, 1, num_layers)
                for _ in range(latent_size)
            ]
        )
        self.fc = MLP(embedding_dim, hidden_dim, hidden_dim, num_layers)

    def forward(self, x, embeddings):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags + 1, step=1).transpose(2, 3)
        # (batch_size, lags+length, hidden_dim)
        embeddings = self.fc(embeddings)
        # batch_embeddings: (batch_size, lags+length, hidden_dim) -> (batch_size, length, lags+1, hidden_dim) -> (batch_size*length, hidden_dim)
        # batch_embeddings = embeddings.unfold(
        #     dimension=1, size=self.lags+1, step=1).transpose(2, 3)[:, :, -1].reshape(batch_size * length, -1)
        batch_embeddings = (
            embeddings[:, -length:]
            .expand(batch_size, length, -1)
            .reshape(batch_size * length, -1)
        )
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1:]  # (batch_size*length, x_dim)
        # (batch_size*length, lags*x_dim)
        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)
            batch_inputs = torch.cat(
                (batch_embeddings, batch_x_lags, batch_x_t[:, :, i]), dim=-1
            )
            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            data_J = vmap(jacfwd(self.gs[i]))(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian
