
import torch
import torch.nn as nn

_loss_fn_class = nn.CrossEntropyLoss()
_loss_fn_orientation = nn.CrossEntropyLoss()

def orion_loss(class_logits, class_labels, orientation_logits, orientation_labels,config: dict) -> nn.Module:
    """
    Calculates the multi-task loss for the ORION model.
    This is a weighted sum of the classification loss and the orientation loss.
    Args:
        class_logits: The raw output from the model's classification head.
        class_labels: The ground truth class labels.
        orientation_logits: The raw output from the model's orientation head.
        orientation_labels: The ground truth orientation labels.
        config (dict): The experiment configuration dictionary, which should
                       contain the gamma hyperparameter.
    Returns:
        torch.Tensor: The final combined loss value.
    """
    # Get the gamma value from the config file, with a sensible default
    gamma = config['training']['gamma']

    class_loss = _loss_fn_class(class_logits, class_labels)
    orientation_loss = _loss_fn_orientation(orientation_logits, orientation_labels)

    total_loss = (1 - gamma) * class_loss + gamma * orientation_loss
    return total_loss


def pointnet_loss(outputs, labels, trans_feat, config):
    """
    Calculates the total loss for PointNet.
    NOTE: This uses CrossEntropyLoss, so the model should output raw logits.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing'])
    base_loss = criterion(outputs, labels)

    # Add feature transformation regularization loss
    if config['model']['feature_transform'] and trans_feat is not None:
        identity = torch.eye(trans_feat.size(-1), device=trans_feat.device).unsqueeze(0)
        # The original paper uses L2 norm
        mat_diff_loss = torch.mean(torch.norm(torch.bmm(trans_feat, trans_feat.transpose(2, 1)) - identity, dim=(1, 2)))
        total_loss = base_loss + 0.001 * mat_diff_loss
        return total_loss
    else:
        return base_loss