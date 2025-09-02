# scripts/train_orion.py
import torch
import yaml
from pathlib import Path
import argparse
# Import all the necessary pieces from your src package
from classification.models.orion import ORION2Layer as orion
from classification.data_utils.modelnet_dataset import VoxelizedOrientedDataset
from classification.engine.train_engine import train_engine
from classification.utils.losses import orion_loss

def orion_step(model, batch, device, config):
    """Performs one forward/backward pass for the ORION model."""
    voxels, class_labels, orientation_labels = batch
    voxels = voxels.to(device).float()
    class_labels = class_labels.to(device)
    orientation_labels = orientation_labels.to(device)

    # Forward pass
    class_logits, orientation_logits = model(voxels)

    # Loss calculation
    loss = orion_loss(
        class_logits, class_labels,
        orientation_logits, orientation_labels,
        config
    )

    # Get predictions for accuracy calculation (we only care about class accuracy here)
    _, predictions = torch.max(class_logits.data, 1)

    return loss, predictions, class_labels


def main():
    # 1. Load the configuration file for this specific experiment
    parser = argparse.ArgumentParser(description='Train ORION model on ModelNet dataset')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to the config file')
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup the specific dataset for PointNet
    print("Setting up PointNet dataset...")
    train_dataset = VoxelizedOrientedDataset(root=config['data']['path'], split='train')
    val_dataset = VoxelizedOrientedDataset(root=config['data']['path'], split='val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Setup the specific model
    print("Creating PointNet model...")
    model = orion(num_classes=config['model']['num_classes'], num_orientations=config['model']['num_orientations'])
    model.to(device)

    # 4. Call the main training engine with the PointNet-specific parts
    print("Starting training engine for PointNet...")
    train_engine(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        train_step_fn=orion_step, # The function specific to PointNet
        val_step_fn=orion_step
    )

if __name__ == '__main__':
    # Example usage: python train_orion.py --config configs/modelnet10/orion.ymal
    main()