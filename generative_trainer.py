import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import json
import copy
import sys
from generative_models import *
import pandas as pd
import os
from os import path

from generative_models import *
from loss import *

RESULTS_DIR = "results"


class GenerativeTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.validate_freq = config.validate_freq

        # get appropriate models from global namespace and instantiate them
        try:
            self.model = eval(self.config.model)(**self.config.__dict__)
        except Exception as e:
            print("An error occurred while creating the model:")
            print(e)
            sys.exit(-1)

        self.loss_fn = eval(self.config.loss_fn)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # directories for saving results
        self.experiment_dir = path.join(RESULTS_DIR, config.exp_name)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.save_config()

    def save_config(
        self,
    ):
        file_name = path.join(self.experiment_dir, "config.json")
        save_config = copy.deepcopy(self.config)
        with open(file_name, "w") as json_file:
            json.dump(vars(save_config), json_file)
        # torch.save(self.config, file_name)

    def load_data(self, X, y):
        # Split the dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False
        )

    def train(self):
        self.model.train()
        best_val_loss = float("inf")
        current_iter = 0
        val_losses = []  # 用于记录每次验证的损失

        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if current_iter % self.validate_freq == 0:
                    val_loss = self.validate()
                    # 记录 epoch, 迭代次数和验证损失
                    val_losses.append(
                        {
                            "epoch": epoch,
                            "iteration": current_iter,
                            "val_loss": val_loss,
                        }
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        file_name = path.join(self.experiment_dir, "best_model.pth")
                        torch.save(self.model.state_dict(), file_name)
                    self.model.train()
                current_iter += 1

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(self.train_loader)}")

        # 训练结束后，将验证损失保存到文件
        file_name = path.join(self.experiment_dir, "accuracies_losses_valid.csv")
        df = pd.DataFrame(val_losses)
        # df = pd.DataFrame(val_losses, columns=["Epoch", "Validation Loss"])
        df.to_csv(file_name, index=False)

        # with open("validation_losses.txt", "w") as f:
        #     for loss in val_losses:
        #         f.write(f"{loss}\n")

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch)
                total_loss += loss.item()
        val_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {val_loss}")
        return val_loss
