import torch
import torch.nn.functional as F


def simclr_anchor_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    extra_pos_mask = None,
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        temperature (float): temperature factor for the loss. Defaults to 0.1.
        extra_pos_mask (Optional[torch.Tensor]): boolean mask containing extra positives other
            than normal across-view positives. Defaults to None.
    Returns:
        torch.Tensor: SimCLR loss.
    """

    # pos = torch.einsum("nc,nc->n", [z1, z2.detach()]).unsqueeze(-1)
    # neg = torch.einsum("nc,ck->nk", [z1, z2.detach()])
    # logits = torch.cat([pos, neg], dim=1)
    # logits /= temperature
    # # targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)

    # targets = torch.range(0, z1.size(0)-1, device=z1.device, dtype=torch.long)
    # loss = F.cross_entropy(neg, targets)


    # return F.cross_entropy(logits, targets)



    device = z1.device

    b = z1.size(0)
    z = torch.cat((z1, z2), dim=0)
    z = F.normalize(z, dim=-1)
    z1 = F.normalize(z1, dim=-1)

    # logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits = torch.einsum("if, jf -> ij", z, z.detach()) / temperature

    # logits = torch.einsum("if, jf -> ij", z1, z) / temperature
    # import ipdb; ipdb.set_trace()


    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    pos_mask[:, b:].fill_diagonal_(True)
    # pos_mask[b:, :].fill_diagonal_(True)

    # if we have extra "positives"
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask, extra_pos_mask)

    # all matches excluding the main diagonal
    logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

    exp_logits = torch.exp(logits) * logit_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)
    # mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # loss
    loss = -mean_log_prob_pos.mean()
    return loss
