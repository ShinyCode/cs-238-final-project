'''
file: splicer.py
--------------------------------------------------------------------------------
Takes pickled data files and creates a plot that overlays both on top of each other.
'''
import os
import sys
import pickle
import plotting
import matplotlib.pyplot as plt

def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    title = sys.argv[3]
    outfile = sys.argv[4]
    data1 = pickle.load(open(os.path.join(os.getcwd(), file1), 'rb'))
    data2 = pickle.load(open(os.path.join(os.getcwd(), file2), 'rb'))
    rewards1 = data1['rewards']
    rewards2 = data2['rewards']
    e1 = plotting.episode2epoch(rewards1, 20)
    e2 = plotting.episode2epoch(rewards2, 20)
    plot_cum_reward(e1, e2, title, outfile)

def plot_cum_reward(rewards1, rewards2, title, filepath):
    r1 = []
    iepisode1 = []
    for _r, _iepisode in rewards1:
        r1.append(_r)
        iepisode1.append(_iepisode)
    r2 = []
    iepisode2 = []
    for _r, _iepisode in rewards2:
        r2.append(_r)
        iepisode2.append(_iepisode)
    g_base, = plt.plot(iepisode1, r1, label='G_base')
    g_ext, = plt.plot(iepisode2, r2, label='G_ext')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.ylim(ymin=-600, ymax=0)
    plt.title(title)
    plt.legend(handles=[g_base, g_ext])
    filename = os.path.join(os.getcwd(), filepath)
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    main()
