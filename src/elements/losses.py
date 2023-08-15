import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.distribution import Distribution

from hparams import get_hparams
from ..utils import scale_pixels


def get_reconstruction_loss():
    params = get_hparams()
    if params.loss_params.reconstruction_loss == 'default':
        return LogProb(data_shape=params.data_params.shape)
    elif params.loss_params.reconstruction_loss == 'mol':
        return DiscMixLogistic(
            data_shape=params.data_params.shape,
            data_num_bits=params.data_params.num_bits,
            num_output_mixtures=params.model_params.num_output_mixtures,
            min_log_scale=params.model_params.min_mol_logscale,
            distribution_base=params.model_params.distribution_base,
            gradient_smoothing_beta=params.model_params.gradient_smoothing_beta,
        )
    elif params.loss_params.reconstruction_loss == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f'Unknown reconstruction loss: {params.loss_params.reconstruction_loss}')


def get_kl_loss():
    params = get_hparams()
    if params.loss_params.kldiv_loss == 'default':
        return KLDivergence(
            distribution_base=params.model_params.distribution_base,
            gradient_smoothing_beta=params.model_params.gradient_smoothing_beta,
            data_shape=params.data_params.shape,
        )
    else:
        raise ValueError(f'Unknown kl loss: {params.loss_params.kldiv_loss}')


class LogProb(nn.Module):
    def __init__(self, data_shape):
        super(LogProb, self).__init__()
        self.data_shape = data_shape

    def forward(self, targets, distribution: Distribution, global_batch_size=32):
        c = self.data_shape[-1] if len(self.data_shape) > 2 else 1
        pixel_count = torch.prod(torch.tensor(self.data_shape))
        targets = targets.reshape(distribution.batch_shape)
        log_probs = distribution.log_prob(targets)

        mean_axis = list(range(1, len(log_probs.size())))
        per_example_loss = torch.sum(log_probs, dim=mean_axis)  # B
        avg_per_example_loss = per_example_loss / (
                np.prod([log_probs.size()[i] for i in mean_axis]) * c)  # B

        assert len(per_example_loss.size()) == len(avg_per_example_loss.size()) == 1

        scalar = global_batch_size * pixel_count

        loss = torch.sum(per_example_loss) / scalar
        # divide by ln(2) to convert to bit range (for visualization purposes only)
        avg_loss = torch.sum(avg_per_example_loss) / (global_batch_size * np.log(2))
        return loss, avg_loss, distribution.mean, distribution.stddev


class DiscMixLogistic(nn.Module):
    def __init__(self,
                 data_shape,
                 data_num_bits,
                 num_output_mixtures,
                 min_log_scale,
                 distribution_base,
                 gradient_smoothing_beta,
                 ):
        super(DiscMixLogistic, self).__init__()
        self.data_shape = data_shape
        self.num_output_mixtures = num_output_mixtures
        self.min_mol_logscale = min_log_scale
        self.distribution_base = distribution_base
        self.gradient_smoothing_beta = gradient_smoothing_beta

        # Only works for when images are [0., 1.] normalized for now
        self.num_classes = 2. ** data_num_bits - 1.
        self.min_pix_value = scale_pixels(0., data_num_bits)
        self.max_pix_value = scale_pixels(255., data_num_bits)

    def forward(self, targets, distribution, global_batch_size=32):
        # targets:  B, C, H, W
        # logits:   B, M * (3 * C + 1), H, W

        logits = distribution.logits
        h, w, c = self.data_shape
        assert len(targets.shape) == 4
        B, C, H, W = targets.size()
        assert C == 3  # only support RGB for now
        n = self.num_output_mixtures
        targets = targets.unsqueeze(2)      # B, C, 1, H, W

        logit_probs = logits[:, :n, :, :]   # B, M, H, W
        l = logits[:, n:, :, :]             # B, M*C*3 ,H, W
        l = l.reshape(B, c, 3 * n, H, W)    # B, C, 3 * M, H, W

        means = l[:, :, :n, :, :]           # B, C, M, H, W
        inv_stdv, log_scales = self._compute_inv_stdv(l[:, :, n: 2 * n, :, :])
        coeffs = torch.tanh(l[:, :, 2 * n: 3 * n, :, :])  # B, C, M, H, W

        # RGB AR
        mean1 = means[:, 0:1, :, :, :]                                # B, 1, M, H, W
        mean2 = means[:, 1:2, :, :, :] \
                + coeffs[:, 0:1, :, :, :] * targets[:, 0:1, :, :, :]  # B, 1, M, H, W
        mean3 = means[:, 2:3, :, :, :] \
                + coeffs[:, 1:2, :, :, :] * targets[:, 0:1, :, :, :] \
                + coeffs[:, 2:3, :, :, :] * targets[:, 1:2, :, :, :]  # B, 1, M, H, W

        means = torch.cat([mean1, mean2, mean3], dim=1)                # B, C, M, H, W
        centered = targets - means  # B, C, M, H, W

        plus_in = inv_stdv * (centered + 1. / self.num_classes)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.num_classes)
        cdf_min = torch.sigmoid(min_in)

        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min  # B, C, M, H, W
        mid_in = inv_stdv * centered
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in this code)
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        broadcast_targets = torch.broadcast_to(targets, size=[B, C, n, H, W])
        log_probs = torch.where(broadcast_targets == self.min_pix_value, log_cdf_plus,
                                torch.where(broadcast_targets == self.max_pix_value, log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(torch.clamp(cdf_delta, min=1e-12)),
                                                        log_pdf_mid - np.log(self.num_classes / 2))))  # B, C, M, H, W

        log_probs = torch.sum(log_probs, dim=1) + F.log_softmax(logit_probs, dim=1)  # B, M, H, W
        negative_log_probs = -torch.logsumexp(log_probs, dim=1)  # B, H, W

        mean_axis = list(range(1, len(negative_log_probs.size())))
        per_example_loss = torch.sum(negative_log_probs, dim=mean_axis)  # B
        avg_per_example_loss = per_example_loss / (
                np.prod([negative_log_probs.size()[i] for i in mean_axis]) * c)  # B

        assert len(per_example_loss.size()) == len(avg_per_example_loss.size()) == 1

        scalar = global_batch_size * h * w * c

        loss = torch.sum(per_example_loss) / scalar
        # divide by ln(2) to convert to bit range (for visualization purposes only)
        avg_loss = torch.sum(avg_per_example_loss) / (global_batch_size * np.log(2))

        return loss, avg_loss, means, log_scales

    def _compute_inv_stdv(self, logits):
        softplus = nn.Softplus(beta=self.gradient_smoothing_beta)
        if self.distribution_base == 'std':
            scales = torch.maximum(softplus(logits),
                                   torch.as_tensor(np.exp(self.min_mol_logscale)))
            inv_stdv = 1. / scales  # Not stable for sharp distributions
            log_scales = torch.log(scales)

        elif self.distribution_base == 'logstd':
            log_scales = torch.maximum(logits, torch.as_tensor(np.array(self.min_mol_logscale)))
            inv_stdv = torch.exp(-self.gradient_smoothing_beta * log_scales)
        else:
            raise ValueError(f'distribution base {self.distribution_base} not known!!')

        return inv_stdv, log_scales


class BernoulliLoss(nn.Module):
    def __init__(self, data_shape):
        super(BernoulliLoss, self).__init__()
        self.data_shape = data_shape

    def forward(self, targets, logits, global_batch_size):
        targets = targets[:, :, 2:30, 2:30]
        logits = logits[:, :, 2:30, 2:30]

        loss_value = Bernoulli(logits=logits)
        recon = loss_value.log_prob(targets)
        mean_axis = list(range(1, len(recon.size())))
        per_example_loss = - torch.sum(recon, dim=mean_axis)
        scalar = global_batch_size * np.prod(self.data_shape)
        loss = torch.sum(per_example_loss) / scalar
        avg_loss = torch.sum(per_example_loss) / global_batch_size
        model_means, log_scales = None, None
        return loss, avg_loss, model_means, log_scales


class KLDivergence(nn.Module):

    def __init__(self, distribution_base, gradient_smoothing_beta, data_shape):
        super(KLDivergence, self).__init__()
        self.distribution_base = distribution_base
        self.gradient_smoothing_beta = gradient_smoothing_beta
        self.data_shape = data_shape

    def forward(self, prior: Distribution, posterior: Distribution, global_batch_size=32):
        if self.distribution_base == 'std':
            loss = self.calculate_std_loss(prior.mean, posterior.mean, prior.stddev, prior.stddev)
        elif self.distribution_base == 'logstd':
            loss = self.calculate_logstd_loss(prior.mean, posterior.mean, prior.stddev, posterior.stddev, self.gradient_smoothing_beta)
        else:
            raise ValueError(f'distribution base {self.distribution_base} not known!!')

        mean_axis = list(range(1, len(loss.size())))
        per_example_loss = torch.sum(loss, dim=mean_axis)
        n_mean_elems = np.prod([loss.size()[a] for a in mean_axis])  # heads * h * w  or h * w * z_dim
        avg_per_example_loss = per_example_loss / n_mean_elems

        assert len(per_example_loss.shape) == 1

        scalar = global_batch_size * np.prod(self.data_shape)
        loss = torch.sum(per_example_loss) / scalar
        # divide by ln(2) to convert to KL rate (average space bits/dim)
        avg_loss = torch.sum(avg_per_example_loss) / (global_batch_size * np.log(2))

        return loss, avg_loss

    @staticmethod
    @torch.jit.script
    def calculate_std_loss(p_mu, q_mu, p_sigma, q_sigma):
        term1 = (p_mu - q_mu) / q_sigma
        term2 = p_sigma / q_sigma
        loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        loss = torch.nan_to_num(loss, nan=0.0)
        return loss

    @staticmethod
    @torch.jit.script
    def calculate_logstd_loss(p_mu, q_mu, p_sigma, q_sigma, gradient_smoothing_beta: float = 1.0):
        q_logstd = q_sigma
        p_logstd = p_sigma

        p_std = torch.exp(gradient_smoothing_beta * p_logstd)
        inv_q_std = torch.exp(gradient_smoothing_beta * q_logstd)

        term1 = (p_mu - q_mu) * inv_q_std
        term2 = p_std * inv_q_std
        loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return loss


class SSIM(nn.Module):
    def __init__(self, image_channels, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        super(SSIM, self).__init__()
        self.max_val = max_val

        self.k1 = k1
        self.k2 = k2
        self.filter_size = filter_size

        self.compensation = 1.

        self.kernel = SSIM._fspecial_gauss(filter_size, filter_sigma, image_channels)

    @staticmethod
    def _fspecial_gauss(filter_size, filter_sigma, image_channels):
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
        coords = torch.arange(0, filter_size, dtype=torch.float32)
        coords -= (filter_size - 1.0) / 2.0

        g = torch.square(coords)
        g *= -0.5 / np.square(filter_sigma)

        g = torch.reshape(g, shape=(1, -1)) + torch.reshape(g, shape=(-1, 1))
        g = torch.reshape(g, shape=(1, -1))  # For tf.nn.softmax().
        g = F.softmax(g, dim=-1)
        g = torch.reshape(g, shape=(1, 1, filter_size, filter_size))
        return torch.tile(g, (image_channels, 1, 1, 1))  # .cuda()  # out_c, in_c // groups, h, w

    def _apply_filter(self, x):
        shape = list(x.size())
        x = torch.reshape(x, shape=[-1] + shape[-3:])  # b , c , h , w
        y = F.conv2d(x, weight=self.kernel, stride=1, padding=(self.filter_size - 1) // 2,
                     groups=x.shape[1])  # b, c, h, w
        return torch.reshape(y, shape[:-3] + list(y.size()[1:]))

    def _compute_luminance_contrast_structure(self, x, y):
        c1 = (self.k1 * self.max_val) ** 2
        c2 = (self.k2 * self.max_val) ** 2

        # SSIM luminance measure is
        # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
        mean0 = self._apply_filter(x)
        mean1 = self._apply_filter(y)
        num0 = mean0 * mean1 * 2.0
        den0 = torch.square(mean0) + torch.square(mean1)
        luminance = (num0 + c1) / (den0 + c1)

        # SSIM contrast-structure measure is
        #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
        # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
        #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
        #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
        num1 = self._apply_filter(x * y) * 2.0
        den1 = self._apply_filter(torch.square(x) + torch.square(y))
        c2 *= self.compensation
        cs = (num1 - num0 + c2) / (den1 - den0 + c2)

        # SSIM score is the product of the luminance and contrast-structure measures.
        return luminance, cs

    def _compute_one_channel_ssim(self, x, y):
        luminance, contrast_structure = self._compute_luminance_contrast_structure(x, y)
        return (luminance * contrast_structure).mean(dim=(-2, -1))

    def forward(self, targets, outputs):
        ssim_per_channel = self._compute_one_channel_ssim(targets, outputs)
        return ssim_per_channel.mean(dim=-1)


class StructureSimilarityIndexMap(nn.Module):
    def __init__(self, image_channels, unnormalized_max=255., filter_size=11):
        super(StructureSimilarityIndexMap, self).__init__()
        self.ssim = SSIM(image_channels=image_channels, max_val=unnormalized_max, filter_size=filter_size)

    def forward(self, targets, outputs, global_batch_size):
        if targets.size() != outputs.size():
            targets = targets.reshape(outputs.size())
        targets = targets * 127.5 + 127.5
        outputs = outputs * 127.5 + 127.5
        assert targets.size() == outputs.size()
        per_example_ssim = self.ssim(targets, outputs)
        mean_axis = list(range(1, len(per_example_ssim.size())))
        per_example_ssim = torch.sum(per_example_ssim, dim=mean_axis)

        loss = torch.sum(per_example_ssim) / global_batch_size
        return loss
