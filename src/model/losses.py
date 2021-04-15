import torch
import torch.nn.functional as F


def focal(
    y_pred,
    y_true,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        y_pred: A float tensor of arbitrary shape.
                The predictions for each example.
        y_true: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    anchor_state = y_true[:, -1]  # -1 for ignore, 0 for background, 1 for object
    indices = torch.where(anchor_state != -1)[0].tolist()

    target = y_true[indices, :-1]
    output = y_pred[indices, :]

    p = torch.sigmoid(output)
    ce_loss = F.binary_cross_entropy_with_logits(
        output, target, reduction="none"
    )
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def smooth_L1(y_pred,
              y_true,
              sigma=3.0,
              alpha=1.0):
    """ Create a smooth L1 loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    sigma_sq = sigma**2

    anchor_state = y_true[:, -1]  # -1 for ignore, 0 for background, 1 for object
    indices = torch.where(anchor_state != -1)[0].tolist()
    if len(indices) == 0:
        return torch.zeros(1, requires_grad=True)

    target = y_true[indices, :-1]
    output = y_pred[indices, :]

    regression_dif = torch.abs(output-target)
    regressions_loss = torch.where(torch.less(regression_dif, 1.0/sigma_sq),
                                   0.5*sigma_sq*torch.pow(regression_dif, 2),
                                   regression_dif - 0.5/sigma_sq)
    return alpha*torch.mean(regressions_loss)


if __name__ == "__main__":
    y_pred = torch.randn(10, 1000, 8)
    y_true = torch.randn(10, 1000, 9)

    focal_loss = focal(y_pred, y_true)
    L1_loss = smooth_L1(y_pred, y_true)