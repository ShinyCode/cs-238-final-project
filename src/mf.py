import numpy as np
import random

# Return GC or NGC
def epsilon_greedy(s, Q, epsilon, a_space):
    if np.random.uniform(0, 1) < epsilon:
        return random.sample(range(len(a_space)), 1)[0]
    return np.argmax(Q[s, :])

def print_info(iepisode, steps, cum_reward, epsilon):
    print "Episode %d:\t steps = %d\tcum_reward = %f\tepsilon = %f" % (iepisode, steps, cum_reward, epsilon)

# Q-learning
def q_learning(env, num_episodes=2000, gamma=0.999, alpha=0.2, epsilon=0.9, decay_rate=0.995):
    Q = np.ones((env._ns(), env._na()))
    rewards = []
    states_seen = set()
    for iepisode in xrange(num_episodes):
        history = []
        done = False
        cum_reward = 0
        s = env._s2i(env._reset())
        while not done:
            states_seen.add(s)
            a = epsilon_greedy(s, Q, epsilon, env.a_space)
            sp, r, done = env._step(a)
            sp = env._s2i(sp)
            cum_reward += r
            history.append((s, a, r, sp, done))
            s = sp
        rewards.append(cum_reward)
        print_info(iepisode, len(history), cum_reward, epsilon)
        epsilon *= decay_rate
        for s, a, r, sp, done in reversed(history):
            if not done:
                Q[s, a] += alpha * (r + gamma * np.max(Q[sp, :]) - Q[s, a])
            else:
                Q[s, a] += alpha * (r - Q[s, a])
    return Q, states_seen

# SARSA
def sarsa(env, num_episodes=2000, gamma=0.999, alpha=0.2, epsilon=0.8, decay_rate=0.995):
    pass

def back_out_policy(Q, states_seen, env):
    states_seen = sorted(list(states_seen))
    policy = np.argmax(Q[states_seen], axis=1)
    policy = np.reshape(policy, (len(policy), 1))
    states_seen = np.reshape(states_seen, (len(states_seen), 1))
    return np.concatenate((env._i2s(states_seen), policy), axis=1)
