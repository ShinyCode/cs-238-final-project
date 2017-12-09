import numpy as np
import random
import copy

# Return GC or NGC
def epsilon_greedy(s, Q, epsilon, a_space):
    if np.random.uniform(0, 1) < epsilon:
        return random.sample(range(len(a_space)), 1)[0]
    return np.argmax(Q[s, :])

def print_info(iepisode, steps, cum_reward, epsilon, dQ):
    print "Episode %d:\t steps = %d\tcum_reward = %f\tepsilon = %f\tdQ = %f" % (iepisode, steps, cum_reward, epsilon, dQ)

# Q-learning
def q_learning(env, num_episodes=10000, gamma=0.999, alpha=0.2, epsilon=0.9, decay_rate=0.995, tolerance=1e-2, max_steps=100):
    Q = np.ones((env._ns(), env._na()))
    rewards = []
    states_seen = set()
    for iepisode in xrange(num_episodes):
        Qprev = copy.deepcopy(Q)
        history = []
        done = False
        cum_reward = 0
        s = env._s2i(env._reset())
        steps = 0
        while not done and steps < max_steps:
            steps += 1
            states_seen.add(s)
            a = epsilon_greedy(s, Q, epsilon, env.a_space)
            sp, r, done = env._step(a)
            sp = env._s2i(sp)
            cum_reward += r
            history.append((s, a, r, sp, done))
            s = sp
        rewards.append(cum_reward)
        for s, a, r, sp, done in reversed(history):
            if not done:
                Q[s, a] += alpha * (r + gamma * np.max(Q[sp, :]) - Q[s, a])
            else:
                Q[s, a] += alpha * (r - Q[s, a])
        dQ = np.sum(np.square(Q - Qprev))
        print_info(iepisode, len(history), cum_reward, epsilon, dQ)
        epsilon *= decay_rate
        if dQ < tolerance:
            break
    return Q, states_seen

# SARSA
def sarsa(env, num_episodes=2000, gamma=0.999, alpha=0.2, epsilon=0.8, decay_rate=0.995):
    Q = np.ones((env._ns(), env._na()))
    rewards = []
    states_seen = set()
    for iepisode in xrange(num_episodes):
        history = []
        done = False
        cum_reward = 0
        curr_s = env._s2i(env._reset())
        curr_a = epsilon_greedy(curr_s, Q, epsilon, env.a_space)
        while not done:
            states_seen.add(curr_s)
            next_s, r, done = env._step(curr_a)
            next_s = env._s2i(next_s)
            cum_reward += r
            next_a = epsilon_greedy(next_s, Q, epsilon, env.a_space)
            history.append((curr_s, curr_a, r, next_s, next_a, done))
            curr_s = next_s
            curr_a = next_a
        rewards.append(cum_reward)
        print_info(iepisode, len(history), cum_reward, epsilon)
        epsilon *= decay_rate
        for transition in history:
            curr_s, curr_a, r, next_s, next_a, done = transition
            if not done:
                Q[curr_s, curr_a] = Q[curr_s, curr_a] + alpha * (r + gamma * Q[next_s,next_a] - Q[curr_s, curr_a])
            else:
                Q[curr_s, curr_a] = Q[curr_s, curr_a] + alpha * (r  - Q[curr_s, curr_a])
    return Q, states_seen

def back_out_policy(Q, states_seen, env):
    states_seen = sorted(list(states_seen))
    actions = list(np.argmax(Q[states_seen], axis=1))
    policy = {}
    for s, a in zip(states_seen, actions):
        policy[env._i2s(s)] = a
    return policy
    # policy = np.reshape(policy, (len(policy), 1))
    # states_seen = np.reshape(states_seen, (len(states_seen), 1))
    # return np.concatenate((env._i2s(states_seen), policy), axis=1)
