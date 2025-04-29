# utils/plots.py
import matplotlib
matplotlib.use("TkAgg")  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import csv
import os
import math

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
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.ax1, self.ax2, self.ax3, self.ax4 = self.axes.flatten()
        plt.show(block=False)
        plt.pause(0.001)  # Allow the plot to render

        # Lines
        self.loss_line_train, = self.ax1.plot([], [], label='Train Loss')
        self.loss_line_val,   = self.ax2.plot([], [], label='Val Loss')
        self.pplx_line_train, = self.ax3.plot([], [], label='Train Perplexity')
        self.pplx_line_val,   = self.ax4.plot([], [], label='Val Perplexity')

        # Axis Titles
        self.ax1.set_title("Train Loss Curve")
        self.ax2.set_title("Val Loss Curve")
        self.ax3.set_title("Train Perplexity Curve")
        self.ax4.set_title("Val Perplexity Curve")

        # Axis Labels
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.set_xlabel("Epoch")
            ax.grid(True)
            ax.legend()

        self.ax1.set_ylabel("Loss")
        self.ax2.set_ylabel("Loss")
        self.ax3.set_ylabel("Perplexity")
        self.ax4.set_ylabel("Perplexity")

    def update(self, epoch, train_loss, val_loss, train_acc=None, val_acc=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc if train_acc is not None else 0)
        self.val_accs.append(val_acc if val_acc is not None else 0)

        # X range
        x = range(1, len(self.train_losses) + 1)

        # update loss
        self.loss_line_train.set_data(x, self.train_losses)
        self.loss_line_val.set_data(x, self.val_losses)

        # update perplexity
        train_pplx = [math.exp(l) for l in self.train_losses]
        val_pplx = [math.exp(l) for l in self.val_losses]
        self.pplx_line_train.set_data(x, train_pplx)
        self.pplx_line_val.set_data(x, val_pplx)

        # 🛠 Fix for matplotlib jank at first epoch
        if len(self.train_losses) == 1:
            for line in [self.loss_line_train, self.loss_line_val, self.pplx_line_train, self.pplx_line_val]:
                line.set_data([1, 2], [line.get_ydata()[0], line.get_ydata()[0]])

        # update view
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.set_xlim(0, max(1, len(self.train_losses)))
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
