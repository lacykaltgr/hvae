import numpy as np
import torch
from torch import tensor, nn, distributions as dist, Size

from hparams import get_hparams
from src.utils import one_hot


dist.distribution.Distribution.set_default_validate_args(False)


def generate_distribution(mu: tensor, sigma: tensor = None, distribution: str = 'normal',
                          sigma_nonlin: str = None, sigma_param: str = None) -> dist.Distribution:
    """
    Generate parameterized distribution

    :param mu: the mean of the distribution
    :param sigma: the standard deviation of the distribution, not needed for mixture of logistics
    :param distribution: 'mixtures_of_logistics', 'normal', 'laplace
    :param sigma_nonlin: 'logstd', 'std'
    :param sigma_param: 'var', 'std'
    :return: torch.distributions.Distribution object
    """
    params = get_hparams().model_params

    if distribution == 'mixture_of_logistics' or distribution == 'mol':
        return MixtureOfLogistics(
            logits=mu,
            n_output_mixtures=params.model_params.n_output_mixtures,
            temperature=1,
            distribution_base=params.model_params.distribution_base,
            gradient_smoothing_beta=params.model_params.output_gradient_smoothing_beta,
            n_channels=params.data_params.shape[-1],
            min_mol_logscale=params.model_params.min_mol_logscale,
            min_pix_value=0,
            max_pix_value=255,
        )

    sigma_nonlin = params.distribution_base if sigma_nonlin is None else sigma_nonlin
    sigma_param = params.distribution_sigma_param if sigma_param is None else sigma_param
    beta = params.gradient_smoothing_beta

    if sigma_nonlin == 'logstd':
        sigma = torch.exp(sigma * beta)
    elif sigma_nonlin == 'std':
        sigma = torch.nn.Softplus(beta=beta)(sigma)
    else:
        raise ValueError(f'Unknown sigma_nonlin {sigma_nonlin}')

    if sigma_param == 'var':
        sigma = torch.sqrt(sigma)
    elif sigma_param != 'std':
        raise ValueError(f'Unknown sigma_param {sigma_param}')

    if distribution == 'normal':
        return dist.Normal(loc=mu, scale=sigma)
    elif distribution == 'laplace':
        return dist.Laplace(loc=mu, scale=sigma)
    else:
        raise ValueError(f'Unknown distr {distribution}')


class MixtureOfLogistics(dist.distribution.Distribution):

    """
    Mixture of logistics distribution

    :param logits: the logits of the distribution
    :param n_output_mixtures: the number of output mixtures
    :param temperature: the temperature of the distribution
    :param n_channels: the number of channels of the distribution
    :param gradient_smoothing_beta: the beta parameter for the gradient smoothing
    :param min_mol_logscale: the minimum logscale of the distribution
    :param min_pix_value: the minimum pixel value of the distribution
    :param max_pix_value: the maximum pixel value of the distribution
    :param distribution_base: 'std', 'logstd'
    """
    def __init__(self,
                 logits: tensor,
                 n_output_mixtures: int = 10,
                 temperature: float = 1.,
                 n_channels: int = 3,
                 gradient_smoothing_beta: int = 1,
                 min_mol_logscale: float = -7.0,
                 min_pix_value: int = 0,
                 max_pix_value: int = 255,
                 distribution_base: str = 'std'):
        super(MixtureOfLogistics, self).__init__()
        self.logits = logits
        self.n_output_mixtures = n_output_mixtures
        self.temperature = temperature
        self.distribution_base = distribution_base
        self.n_channels = n_channels
        self.gradient_smoothing_beta = gradient_smoothing_beta
        self.min_pix_value = min_pix_value
        self.max_pix_value = max_pix_value
        self.min_mol_logscale = min_mol_logscale

    def sample(self, sample_shape: Size = Size()) -> tensor:
        B, _, H, W = self.logits.size()  # B, M*(3*C+1), H, W,
        n = self.n_output_mixtures
        t = self.temperature

        logit_probs = self.logits[:, :n, :, :]  # B, M, H, W
        l = self.logits[:, n:, :, :]  # B, M*C*3 ,H, W
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

        args = dict(min=self.min_pix_value, max=self.max_pix_value)
        # Autoregressive prediction of RGB
        x0 = torch.clamp(x[:, 0:1, :, :], **args)  # B, 1, H, W
        x1 = torch.clamp(x[:, 1:2, :, :] + coeffs[:, 0:1, :, :] * x0, **args)  # B, 1, H, W
        x2 = torch.clamp(x[:, 2:3, :, :] + coeffs[:, 1:2, :, :] * x0 + coeffs[:, 2:3, :, :] * x1, **args)  # B, 1, H, W
        x = torch.cat([x0, x1, x2], dim=1)  # B, C, H, W
        return x

    def _compute_scales(self, logits):
        if self.distribution_base == 'std':
            scales = torch.maximum(
                nn.Softplus(beta=self.gradient_smoothing_beta)(logits),
                torch.as_tensor(np.exp(self.min_mol_logscale)))
        elif self.distribution_base == 'logstd':
            log_scales = torch.maximum(
                logits, torch.as_tensor(np.array(self.min_mol_logscale)))
            scales = torch.exp(self.gradient_smoothing_beta * log_scales)
        else:
            raise ValueError(f'distribution base {self.distribution_base} not known!!')
        return scales

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def entropy(self) -> torch.Tensor:
        raise NotImplementedError()
