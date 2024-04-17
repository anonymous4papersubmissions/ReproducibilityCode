import os

import torch
import torch.nn.functional as F

def masked_logsumexp(x, mask, eps=1e-8):
    mask_1 = mask.sum(dim=-1)
    m = torch.max(x * mask - ((1 - mask) / eps), dim=-1, keepdim=True).values
    lse = m.squeeze(-1) * mask_1 + torch.log(torch.sum((torch.exp(x - m) * mask), dim=-1) + eps)
    return lse

class NTXent(torch.nn.Module):
    def __init__(self, temperature=1.0):
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation
        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = F.normalize(features, dim=-1)
        device = (torch.device(
                    f'cuda:{os.environ["CUDA_VISIBLE_DEVICE"]}' if \
                    "CUDA_VISIBLE_DEVICE" in os.environ else 'cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # if self.contrast_mode == 'one':
        #     anchor_feature = features[:, 0]
        #     anchor_count = 1
        # elif self.contrast_mode == 'all':
        #     anchor_feature = contrast_feature
        #     anchor_count = contrast_count
        # else:
        #     raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print(logits_mask.shape, logits_mask)
        # print(torch.log(exp_logits.sum(1, keepdim=True)))
        # print(logits[torch.log(exp_logits.sum(1, keepdim=True)[:,0]) < -1e12])
        # print(exp_logits[torch.log(exp_logits.sum(1, keepdim=True)[:,0]) < -1e12])
        # print(exp_logits.sum(1, keepdim=True)[torch.log(exp_logits.sum(1, keepdim=True)[:,0]) < -1e12])
        # print(torch.log(exp_logits.sum(1, keepdim=True))[torch.log(exp_logits.sum(1, keepdim=True)[:,0]) < -1e12])
        # print(logits_mask[torch.log(exp_logits.sum(1, keepdim=True)[:,0]) < -1e12])
        # assert False
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # log_prob = logits - masked_logsumexp(logits, logits_mask)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-12)
        # print(torch.isnan(logits).any(), logits.min(), logits.max())
        # print(torch.isnan(exp_logits).any(), exp_logits.min(), exp_logits.max())
        # print(torch.isnan(torch.log(exp_logits.sum(1, keepdim=True))).any(),
        #                   torch.log(exp_logits.sum(1, keepdim=True).min()),
        #                   torch.log(exp_logits.sum(1, keepdim=True).max()))
        # print(torch.isnan(log_prob).any(), log_prob.min(), log_prob.max())
        # print(torch.isnan(mean_log_prob_pos).any(), mean_log_prob_pos.min(), mean_log_prob_pos.max())
        # print(torch.isnan(mask).any(), mask.max())
        # print(torch.isnan(mean_log_prob_pos).any(), mean_log_prob_pos.max())
        # assert False

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        # print(mean_log_prob_pos.max(), mean_log_prob_pos.min(), log_prob, loss)
        return loss

class SCARF:
    def __init__(self, n_views=2, corruption_rate=0.5):
        self.n_views = n_views
        self.corruption_rate = corruption_rate

    def __call__(self, anchor, random_sample):
        batch_size, m = anchor.size()
        anchor_repeated = anchor.repeat(self.n_views-1, 1, 1)
        corruption_len = int(self.corruption_rate * m)

        # 1: create a mask of size (batch size, m) where for each sample we set the
        # jth column to True at random, such that corruption_len / m = corruption_rate
        # 3: replace x_1_ij by x_2_ij where mask_ij is true to build x_corrupted
        corruption_mask_li = []
        for _ in range(self.n_views-1):
            corruption_mask = torch.zeros_like(anchor, dtype=torch.bool, device=anchor.device)
            for i in range(batch_size):
                corruption_idx = torch.randperm(m)[:corruption_len]
                corruption_mask[i, corruption_idx] = True
            corruption_mask_li.append(corruption_mask)
        corruption_mask = torch.stack(corruption_mask_li, dim=0)

        positive = torch.where(corruption_mask, random_sample, anchor_repeated)
        return anchor, positive