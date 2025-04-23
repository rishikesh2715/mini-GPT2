# utils/plots.py

import matplotlib.pyplot as plt
import csv
import os

class LiveMetricsPlot:
    def __init__(self, save_log_path="results/metrics_log.csv"):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # CSV Logging Setup
        self.log_path = save_log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

        # Plot setup
        plt.style.use("dark_background")
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
        self.loss_line_train, = self.ax1.plot([], [], label='Train Loss')
        self.loss_line_val,   = self.ax1.plot([], [], label='Val Loss')
        self.acc_line_train,  = self.ax2.plot([], [], label='Train Acc')
        self.acc_line_val,    = self.ax2.plot([], [], label='Val Acc')

        for ax in (self.ax1, self.ax2):
            ax.legend()
            ax.grid(True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        self.ax1.set_title("Loss Curve")
        self.ax1.set_ylabel("Loss")
        self.ax2.set_title("Accuracy Curve")
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("Accuracy")

    def update(self, epoch, train_loss, val_loss, train_acc=None, val_acc=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc if train_acc is not None else 0)
        self.val_accs.append(val_acc if val_acc is not None else 0)

        # Update plots
        x = range(1, epoch + 2)
        self.loss_line_train.set_data(x, self.train_losses)
        self.loss_line_val.set_data(x, self.val_losses)
        self.acc_line_train.set_data(x, self.train_accs)
        self.acc_line_val.set_data(x, self.val_accs)

        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

        # Log to CSV
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, self.train_accs[-1], self.val_accs[-1]])

    def save(self, path="results/metrics_plot.png"):
        plt.ioff()
        self.fig.tight_layout()
        self.fig.savefig(path)
        print(f"📊 Saved final metrics plot to {path}")
