import numpy as np
import torch
from torch import tensor, nn

from hparams import get_hparams
from src.utils import one_hot


def get_output_sampler():
    params = get_hparams()
    if params.model_params.output_distribution == 'normal':
        return GaussianSampler(
            distribution_base=params.model_params.distribution_base,
            output_distribution=params.model_params.output_distribution
        )
    elif params.model_params.output_distribution == 'mol':
        return MixtureOfLogisticsSampler(
            n_output_mixtures=10, temperature = 1,
            distribution_base=params.model_params.distribution_base,
            output_distribution=params.model_params.output_distribution,
            output_gradient_smoothing_beta=params.model_params.output_gradient_smoothing_beta,
            n_channels=params.data_params.channels,
            min_mol_logscale=params.loss_params.min_mol_logscale,
        )
    else:
        raise ValueError(f'Unknown output sampler: {params.model_params.output_sampler}')


class GaussianSampler(nn.Module):
    def __init__(self,distribution_base='std', output_distribution='normal', gradient_smoothing_beta=1):
        super(GaussianSampler, self).__init__()
        self.distribution_base = distribution_base
        self.output_distribution = output_distribution
        self.gradient_smoothing_beta = gradient_smoothing_beta

    def forward(self, mean, std,
                temperature=None, projection=None):
        if projection is not None:
            x = torch.cat([mean, std], dim=1)
            x = self.projection(x)
            mean, std = torch.chunk(x, chunks=2, dim=1)

        if self.distribution_base == 'std':
            std = torch.nn.Softplus(beta=self.gradient_smoothing_beta)(std)
        elif self.distribution_base == 'logstd':
            std = torch.exp(self.gradient_smoothing_beta * std)

        #TODO: var vagy std különbség? gyököt vonni vagy nem?

        #if temperature is not None:
        #    std = std * temperature
        z = mean
        if self.output_distribution == 'normal':
            z = self.calculate_z_normal(mean, std)
        elif self.output_distribution == 'laplace':
            z = self.calculate_z_uniform(mean, std)
        return z

    @staticmethod
    @torch.jit.script
    def calculate_z_normal(mean, std):
        eps = torch.empty_like(mean, device=torch.device('cpu')).normal_(0., 1.)
        z = eps * std + mean
        return z

    @staticmethod
    @torch.jit.script
    def calulate_z_uniform(mean, std):
        eps = torch.empty_like(mean, device=torch.device('cpu')).uniform_(0.)
        z = eps * std + mean
        return z


class MixtureOfLogisticsSampler(nn.Module):

    def __init__(self, n_output_mixtures=10, temperature = 1,
                 n_channels = 3, output_gradient_smoothing_beta=1, min_mol_logscale=-7.0,
                 distribution_base='std', output_distribution='normal'):
        super(MixtureOfLogisticsSampler, self).__init__()
        self.n_output_mixtures = n_output_mixtures
        self.temperature = temperature
        self.distribution_base = distribution_base
        self.output_distribution = output_distribution
        self.n_channels = n_channels
        self.output_gradient_smoothing_beta = output_gradient_smoothing_beta
        self.min_mol_logscale = min_mol_logscale

    def forward(self, logits):
        return self.sample(logits)

    def sample(self, logits: tensor, min_pix_value: int = 0, max_pix_value: int = 255) -> tensor:

        B, _, H, W = logits.size()  # B, M*(3*C+1), H, W,
        n = self.n_output_mixtures
        t = self.temperature

        logit_probs = logits[:, :n, :, :]  # B, M, H, W
        l = logits[:, n:, :, :]  # B, M*C*3 ,H, W
        l = l.reshape(B, self.n_channels, 3 * n, H, W)  # B, C, 3 * M, H, W

        model_means = l[:, :, :n, :, :]  # B, C, M, H, W
        scales = self._compute_scales(l[:, :, n: 2 * n, :, :])  # B, C, M, H, W
        model_coeffs = torch.tanh(l[:, :, 2 * n: 3 * n, :, :])  # B, C, M, H, W

        # Gumbel-max to select the mixture component to use (per pixel)
        gumbel_noise = -torch.log(-torch.log(
            torch.Tensor(logit_probs.size()).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W
        logit_probs = logit_probs / t + gumbel_noise
        lambda_ = one_hot(torch.argmax(logit_probs, dim=1), logit_probs.size()[1], dim=1)  # B, M, H, W

        lambda_ = lambda_.unsqueeze(1)  # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(model_means * lambda_, dim=2)  # B, C, H, W
        scales = torch.sum(scales * lambda_, dim=2)  # B, C, H, W
        coeffs = torch.sum(model_coeffs * lambda_, dim=2)  # B, C,  H, W

        # Samples from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).cuda()
        x = means + scales * t * (torch.log(u) - torch.log(1. - u))  # B, C,  H, W

        # Autoregressive prediction of RGB
        x0 = torch.clamp(x[:, 0:1, :, :],
                         min=min_pix_value,
                         max=max_pix_value)  # B, 1, H, W
        x1 = torch.clamp(x[:, 1:2, :, :] + coeffs[:, 0:1, :, :] * x0,
                         min=min_pix_value,
                         max=max_pix_value)  # B, 1, H, W
        x2 = torch.clamp(x[:, 2:3, :, :] + coeffs[:, 1:2, :, :] * x0 + coeffs[:, 2:3, :, :] * x1,
                         min=min_pix_value,
                         max=max_pix_value)  # B, 1, H, W
        x = torch.cat([x0, x1, x2], dim=1)  # B, C, H, W
        return x

    def _compute_scales(self, logits):
        softplus = nn.Softplus(beta=self.output_gradient_smoothing_beta)
        if self.output_distribution_base == 'std':
            scales = torch.maximum(
                softplus(logits), torch.as_tensor(np.exp(self.min_mol_logscale)))
        elif self.output_distribution_base == 'logstd':
            log_scales = torch.maximum(
                logits, torch.as_tensor(np.array(self.min_mol_logscale)))
            scales = torch.exp(self.output_gradient_smoothing_beta * log_scales)
        else:
            raise ValueError(f'distribution base {self.output_distribution_base} not known!!')
        return scales

