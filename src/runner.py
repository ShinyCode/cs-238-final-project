'''
file: runner.py
--------------------------------------------------------------------------------
Runs methods from mf.py on garbage collection MDPs, and also provides methods
for saving parameters and data.
'''
import g_base
import g_ext
import mf
import usage_pattern as up
import plotting
import constants as k
import os
import sys
import pickle

def main():
    env = g_base.GarbageCollectionEnv_Base(100, up.usage_pattern_flat(100, 5))
    Q, indices_seen, rewards, rl_params = mf.q_learning(env, num_episodes=5000, max_steps=1000, epsilon=0.9999, tolerance=1e-5)
    policy = mf.back_out_policy(Q, indices_seen, env)
    print policy

def save_params(title, pattern, final_reward, mdp_params, rl_params):
    outfile = os.path.join(os.getcwd(), title)
    with open(outfile, 'w') as f:
        f.write('pattern: %s\n' % pattern)
        f.write('final_reward: %f\n' % final_reward)
        for p in mdp_params:
            f.write('%s: %s\n' % (p, mdp_params[p]))
        f.write('---------------------------------\n')
        for p in rl_params:
            f.write('%s: %s\n' % (p, rl_params[p]))

def save_data(title, rewards, policy):
    data = {'rewards': rewards, 'policy': policy}
    filename = os.path.join(os.getcwd(), title)
    pickle.dump(data, open(filename, 'wb'))

if __name__ == '__main__':
    main()
