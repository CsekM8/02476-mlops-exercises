import argparse
import copy
import math
import sys
import time
import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import numpy as np

from src.data.data import mnist
from src.models.model import MyAwesomeModel


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.001)
        # add any additional argument that you want
        parser.add_argument("--epoch", default=20)
        parser.add_argument("--batch", default=64)
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        trainloader = torch.utils.data.DataLoader(
            train_set, args.batch, True, num_workers=2
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), args.lr)
        epochs = args.epoch

        since = time.time()

        train_loss_history = []
        train_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        best_acc = 0.0

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.train()

        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs - 1}")
            print("-" * 20)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(trainloader.dataset)
            epoch_acc = torch.true_divide(running_corrects, len(trainloader.dataset))

            print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if best_acc < epoch_acc:
                best_acc = epoch_acc
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

        print()
        time_elapsed = time.time() - since
        print(
            f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best training loss: {best_loss}")
        print(f"Best training accuracy: {best_acc}")

        # save weights
        if not os.path.exists("Weights"):
            os.makedirs("Weights")
        torch.save(best_model_wts, "Weights/best_model_weights")

        train_acc_history = [h.cpu().numpy() for h in train_acc_history]

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.set_title("Training Loss vs. Number of Training Epochs")
        ax1.set_xlabel("Training Epochs")
        ax1.set_ylabel("Training Loss")
        ax1.plot(range(1, epochs + 1), train_loss_history)
        ax1.set_ylim((0, 1.0))
        ax1.set_xticks(np.arange(1, epochs + 1, 1.0))
        ax2.set_title("Training Accuracy vs. Number of Training Epochs")
        ax2.set_xlabel("Training Epochs")
        ax2.set_ylabel("Training Accuracy")
        ax2.plot(range(1, epochs + 1), train_acc_history)
        ax2.set_ylim((0, 1.0))
        ax2.set_xticks(np.arange(1, epochs + 1, 1.0))
        plt.show()

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Eval arguments")
        parser.add_argument("load_model_from", default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        _, test_set = mnist()
        testloader = torch.utils.data.DataLoader(test_set, 64, True, num_workers=2)
        criterion = nn.CrossEntropyLoss()

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.eval()

        running_corrects = 0.0
        running_loss = 0.0

        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        overall_loss = running_loss / len(testloader.dataset)
        overall_accuracy = torch.true_divide(running_corrects, len(testloader.dataset))

        print()
        print(f"Evaluation Loss: {overall_loss}")
        print(f"Evaluation Accuracy:: {overall_accuracy}")


if __name__ == "__main__":
    TrainOREvaluate()
