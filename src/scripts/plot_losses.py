import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curves(logfile, save_path):
    epochs = np.arange(0, len(logfile))
    fig, axes = plt.subplots(3, 1, figsize=(15, 5))
    for ii, loss in enumerate(logfile.columns):
        # smoothen out loss using EMA
        smoothened_losses = logfile[loss].rolling(window=100, min_periods=5, center=True).mean()
        axes[ii].plot(epochs, smoothened_losses, label=loss)
        axes[ii].set_xlabel('iterations')
        axes[ii].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

if __name__ == '__main__':
    path_to_logfile = sys.argv[1]
    bs = 4 # batch size
    save_path = f"./viz_outputs/loss_curves/{os.path.dirname(path_to_logfile).split('.cache_')[-1].replace('/', '_')}_losscurve.png"
    # path_to_logs = '../diffusion-classifier-fewshot/.cache_mucti_v1/colorectal/gamma=0.005/logs/loss_logs.csv'
    logfile = pd.read_csv(path_to_logfile)
    # sum and avg across batch sizes
    # logfile_epoch = {loss : [] for loss in logfile.columns}
    # for i in range(0, len())
    plot_loss_curves(logfile, save_path)

