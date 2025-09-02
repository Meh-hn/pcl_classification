
import torch
import torch.optim as optim
from classification.utils.logging import CsvLogger

from tqdm import tqdm 
import os
import wandb


def get_optimizer(model, config):
    """Creates an optimizer based on the configuration."""
    if config['training']['optimizer'].lower() == 'adamw':
        print("Using AdamW optimizer.")
        return optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    else: # Default to Adam
        print("Using Adam optimizer.")
        return optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

def get_scheduler(optimizer, config):
    """Creates a learning rate scheduler based on the configuration."""
    if config['scheduler']['type'].lower() == 'plateau':
        print("Using ReduceLROnPlateau scheduler.")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # We monitor validation accuracy, so we want to maximize it
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience'],
            verbose=True
        )
    else: # Default to CosineAnnealingLR
        print("Using CosineAnnealingLR scheduler.")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'], # The number of epochs to complete one cosine cycle
            eta_min=1e-6 # Minimum learning rate
        )

def train_engine(config, model, train_loader, val_loader, device, train_step_fn, val_step_fn):
    """
    The main UNIVERSAL training and validation loop.
    It uses model-specific step functions to handle the details.
    """
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    # --- Variables for Tracking and Resuming ---
    start_epoch = 0
    best_val_accuracy = 0.0
    wandb.init(project=config['experiment_name'], config=config)  # Initialize Weights & Biases

    # --- Check for a checkpoint to resume training ---
    resume_checkpoint_path = os.path.join(config['output_dir'], config['experiment_name'], 'last_checkpoint.pth')
    if config['training']['resume_training'] and os.path.exists(resume_checkpoint_path):
        print(f"Resuming training from {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
        best_val_accuracy = checkpoint['best_val_accuracy']

        print(f"Resumed from Epoch {start_epoch}. Best accuracy so far: {best_val_accuracy:.2f}%")
    else:
        print("Starting training from scratch.")

    # --- Logger Setup ---
    log_header = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'learning_rate']
    log_file_path = os.path.join(config['output_dir'], config['experiment_name'], 'training_log.csv')
    csv_logger = CsvLogger(log_file_path, header=log_header)

    # --- Main Training Loop ---
    for epoch in range(start_epoch, config['training']['epochs']):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # The loop is now generic!
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Train")):
            loss, predictions, labels = train_step_fn(model, batch, device, config)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += labels.size(0)
            train_correct += (predictions == labels).sum().item()

            if i % 100 == 99:
              wandb.log({'train_loss/batch': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                loss, predictions, labels = val_step_fn(model, batch, device, config)
                val_loss += loss.item()
                val_total += labels.size(0)
                val_correct += (predictions == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # --- Checkpointing ---
        # 1. Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(config['output_dir'], config['experiment_name'], 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"--> New best model saved to {best_model_path} with accuracy: {val_accuracy:.2f}%")

        # 2. Save a checkpoint for resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'config': config
        }, resume_checkpoint_path)

        # 3. save the loss and accuracy values for logging
        metrics_to_log = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        csv_logger.log(metrics_to_log)

        wandb.log({
        "train/epoch_loss": avg_train_loss,
        "train/epoch_accuracy": train_accuracy,
        "val/epoch_loss": avg_val_loss,
        "val/epoch_accuracy": val_accuracy,
        "epoch": epoch + 1 # Log the epoch number itself
    })
        
    wandb.finish()
    print('Finished Training.')