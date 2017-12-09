import matplotlib.pyplot as plt
import numpy as np
import os

def plot_cum_reward(rewards, title, filepath):
    r = []
    iepisode = []
    for _r, _iepisode in rewards:
        r.append(_r)
        iepisode.append(_iepisode)
    plt.plot(iepisode, r)
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.ylim(ymin=-600, ymax=0)
    plt.title(title)
    filename = os.path.join(os.getcwd(), filepath)
    plt.savefig(filename)
    plt.show()

def episode2epoch(rewards, epoch_size):
    result = []
    epoch = []
    iepoch = 0
    for r, iepisode in rewards:
        epoch.append(r)
        if len(epoch) == epoch_size:
            r_avg = np.mean(epoch)
            result.append((r_avg, iepoch))
            epoch = []
            iepoch += 1
    return result
