import garbage_collection_h0 as gcb
import mf
import usage_pattern as up
import plotting
import constants as k
import os
import sys
import pickle

def main():
    if len(sys.argv) != 1 and len(sys.argv) != 4:
        raise ValueError
    # env = gcb.GarbageCollectionEnv_H0(100, up.usage_pattern_square(100, 0, 5, 10, 5))
    # env = gcb.GarbageCollectionEnv_H0(100, up.usage_pattern_flat(100, 5))
    # env = gcb.GarbageCollectionEnv_H0(100, up.usage_pattern_sawtooth(100, 5, 0, 10))
    env = gcb.GarbageCollectionEnv_H0(100, up.usage_pattern_random(100, 0, 10))
    Q, states_seen, rewards, rl_params = mf.q_learning(env, num_episodes=5000, max_steps=1000, epsilon=0.9999, tolerance=1e-5)
    #Q, states_seen = mf.sarsa(env)
    policy = mf.back_out_policy(Q, states_seen, env)
    print policy
    plotting.plot_cum_reward(plotting.episode2epoch(rewards, 20), 'Performance of Garbage Collector 0', sys.argv[1])
    filename = sys.argv[2]
    save_params(filename, 'random_100_0_10', rewards[-1][0], k.get_mdp_params(), rl_params)
    filename = sys.argv[3]
    save_data(filename, rewards, policy)
    # return policy

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
