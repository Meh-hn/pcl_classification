# scripts/train_pointnet.py
import torch
import yaml
from pathlib import Path
import argparse
# Import all the necessary pieces from your src package
from torchvision import transforms

from classification.models.pointnet import PointNet
from classification.data_utils.modelnet_dataset import PointCloudDataset
from classification.engine.train_engine import train_engine
from classification.utils.losses import pointnet_loss
from classification.utils.pcl_transforms import PointCloudToTensor, RandomRotation, RandomJitter

def pointnet_step(model, batch, device, config):
    """Performs one forward/backward pass for the PointNet model."""
    points, labels = batch
    points = points.transpose(2, 1).to(device).float()
    labels = labels.to(device)

    # Forward pass
    logits, trans_feat = model(points)

    # Loss calculation
    # NOTE: You need to have a pointnet_loss function defined elsewhere
    loss = pointnet_loss(logits, labels, trans_feat, config)

    # Get predictions for accuracy calculation
    _, predictions = torch.max(logits.data, 1)

    return loss, predictions, labels


def main():
    # 1. Load the configuration file for this specific experiment
    parser = argparse.ArgumentParser(description='Train PointNet model on ModelNet dataset')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to the config file')
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
 
    if config['training']['augmentation']:
      train_transforms = transforms.Compose([
      RandomRotation(axis='z', angle_range=(-180, 180)),
      RandomJitter(sigma=0.01, clip=0.05),
      PointCloudToTensor()
      ])
    else: train_transforms = None

    # 2. Setup the specific dataset for PointNet
    print("Setting up PointNet dataset...")
    
    train_dataset = PointCloudDataset(root=config['data']['path'], split='train', n_points=config['data']['num_points'], validation_split=config['data']['validation_split'], transform=train_transforms)
    val_dataset = PointCloudDataset(root=config['data']['path'], split='val', n_points=config['data']['num_points'],validation_split=config['data']['validation_split'], transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 3. Setup the specific model
    print("Creating PointNet model...")
    model = PointNet(num_classes=config['model']['num_classes'], feature_transform=config['model']['feature_transform'])
    model.to(device)

    # 4. Call the main training engine with the PointNet-specific parts
    print("Starting training engine for PointNet...")
    train_engine(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        train_step_fn=pointnet_step, # The function specific to PointNet
        val_step_fn=pointnet_step
    )

if __name__ == '__main__':
    # Example usage: python scripts/train_pointnet.py --config configs/modelnet40/pointnet.ymal
    main()