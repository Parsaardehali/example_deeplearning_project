from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import torch
from loss import DiceLoss
import numpy as np
import os
def train(model,
          train_loader, 
          test_loader, 
          criterion, 
          optimizer, 
          device, 
          num_epochs, 
          training_name, 
          log_interval, 
          checkpoint_interval, 
          save_model_path):
    """
    Training function with visualization every 10 epochs.
    
    Args:
        model: The model to train
        train_loader: DataLoader containing the training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run the training on
        num_epochs: Number of epochs to train for
    """
    # initialize Weights & Biases
    wandb.init(project="Example project",
        name = training_name,config={
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": train_loader.batch_size,
        "epochs": num_epochs,
        "model": model.__class__.__name__})
    model.train()
    acc_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            acc_loss.append(loss.item())
            
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        wandb.log({"epoch_loss": epoch_loss}, step= epoch)
        # Visualize results every 10 epochs
        if (epoch + 1) % log_interval== 0:
            model.eval()
            with torch.no_grad():
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                sample_images, sample_labels = next(iter(test_loader))
                sample_images = sample_images[0].unsqueeze(0).to(device) # add batch dimension
                sample_labels = sample_labels[0].unsqueeze(0).to(device) # add batch dimension
                sample_outputs = model(sample_images)
                ax[0].imshow(sample_images.cpu().squeeze(), cmap='gray')
                ax[0].set_title('Input Image')
                ax[1].imshow(sample_labels.cpu().squeeze(), cmap='gray')
                ax[1].set_title('Ground Truth')
                ax[2].imshow(sample_outputs.cpu().squeeze(), cmap='gray')
                ax[2].set_title('Predicted Mask')
                for a in ax:
                    a.axis('off')
                plt.tight_layout()
                # log the figure to wandb
                wandb.log({"results": wandb.Image(fig)}, step = epoch)
                plt.close(fig)
            model.train()  # Set back to training mode
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            # mkdir if not exists
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            # Save model checkpoint
            checkpoint_path = f"{save_model_path}/{training_name}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")
