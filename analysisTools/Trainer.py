import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import sys
import math


class Trainer:
    """
    Trainer class to manage training and evaluation of PyTorch models.
    """

    def __init__(self, model, lr, epochs, batch_size, device='cpu',isClassification=True):
        self.model = model.to(device)  # Ensure model is on the correct device
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr) 
        self.optimizer =torch.optim.Adam(self.model.parameters(), lr=lr)
        self.isClassification = isClassification
        if self.isClassification:
            self.criterion = nn.CrossEntropyLoss().to(device)  # Move criterion to the device
        else:
            self.criterion = nn.MSELoss().to(device)

    def train_all(self, dataloader):
        """
        Train the model over all epochs, handling data loading and progress display.
        """
        total_batches = len(dataloader)  # Total batches in the dataloader
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # Move data to the device
                self.train_one_epoch(batch_idx, inputs, targets, epoch, total_batches)
            print()  # New line for clean separation of epochs

    def train_one_epoch(self, batch_idx, inputs, targets, epoch, total_batches):
        """
        Perform one epoch of training, process a single batch of data.
        """
        self.model.train()  # Set model to training mode
        self.optimizer.zero_grad() 
        outputs = self.model(inputs)
        outputs = outputs # Remove unnecessary dimensions
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        # Zero the gradients to prepare for backpropagation
        # Display progress with a custom progress bar
        progress = (batch_idx + 1) / total_batches
        bar_length = 30
        block = int(round(bar_length * progress))
        text = "\r[{}] {:.0f}% - Batch {}/{} - Loss: {:.4f}".format(
            "#" * block + "-" * (bar_length - block), progress * 100, batch_idx + 1, total_batches, loss.item())
        sys.stdout.write(text)
        sys.stdout.flush()

    def predict_torch(self, dataloader):
        """
        Use the trained model to predict labels on a new dataset using no gradient calculation for efficiency.
        """
        self.model.eval()  # Set model to evaluation mode
        pred_labels = []
        with torch.no_grad():  # Disable gradient calculation for efficiency
            for batch in dataloader:
                inputs = batch[0].to(self.device)  # Move inputs to the device
                outputs = self.model(inputs)
                if self.isClassification:
                    preds = torch.argmax(outputs, dim=1)  # Get the index of the max log-probability
                else:
                    preds = outputs

                pred_labels.append(preds)

        pred_labels = torch.cat(pred_labels)
        return pred_labels

    def fit(self, training_data, training_labels):
        train_tensor = torch.tensor(training_data, dtype=torch.float32).to(self.device)
        if self.isClassification:
            label_tensor = torch.from_numpy(training_labels).long().to(self.device)
        else:
            label_tensor = torch.tensor(training_labels, dtype=torch.float32).reshape(-1, 1).to(self.device)
        train_dataset = TensorDataset(train_tensor, label_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)  # Train the model on the DataLoader
        output = self.model(train_tensor)  # This will give you the logits
        if self.isClassification:
            predictions = torch.argmax(output, dim=1)
        else:
    # For regression, just use the raw outputs (potentially squeeze if needed)
            predictions = output
        return predictions.detach().cpu().numpy()  # Return predictions as numpy array

    def predict(self, test_data):
        """
        Predict function that serves as an interface between numpy and PyTorch for test data prediction.
        """
        test_tensor = torch.tensor(test_data, dtype=torch.float32).to(self.device)
        test_dataset = TensorDataset(test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        pred_labels = self.predict_torch(test_dataloader)

        return pred_labels.cpu().numpy()  # Return predictions as numpy array