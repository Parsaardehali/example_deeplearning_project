import torch
from torch import optim
from torch.utils.data import random_split
from models import MyUNet
from datasets import MyDataset
from loss import DiceLoss
from trainer import train
from metrics import calculate_dice_score
import json
import os
from argparse import ArgumentParser

def run_training(params):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = params.get("batch_size")
    num_epochs = params.get("num_epochs")
    learning_rate = params.get("learning_rate")
    training_name = params.get("training_name")
    log_interval = params.get("log_interval", 1)
    save_model_path = params.get("save_model_path", "./models/")
    checkpoint_interval = params.get("checkpoint_interval", 5)
    # Load datasets 
    train_dataset = MyDataset(path=params.get("train_data_path"))
    test_dataset = MyDataset(path=params.get("test_data_path"))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = MyUNet(in_channels=1, out_channels=1).to(device)

    criterion = MyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    
    train(model, 
          train_loader,
          test_loader,
          criterion,
          optimizer, 
          device, 
          num_epochs, 
          training_name, 
          log_interval=log_interval,
          checkpoint_interval=checkpoint_interval,
          save_model_path=save_model_path)


# Main execution
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    json_path = parser.parse_args().config
    with open(json_path, "r") as f:
        params = json.load(f)
    run_training(params)
    
