'''
file: mf.py
--------------------------------------------------------------------------------
Provides model-free RL methods, as well as utility functions to back out policies.
'''
import numpy as np
import random
import copy

def epsilon_greedy(s, Q, epsilon, a_space):
    if np.random.uniform(0, 1) < epsilon:
        return random.sample(range(len(a_space)), 1)[0]
    return np.argmax(Q[s, :])

def print_info(iepisode, steps, cum_reward, epsilon, dQ):
    print "Episode %d:\t steps = %d\tcum_reward = %f\tepsilon = %f\tdQ = %f" % (iepisode, steps, cum_reward, epsilon, dQ)

def q_learning(env, num_episodes=5000, gamma=0.999, alpha=0.2, epsilon=0.9999, decay_rate=0.995, tolerance=1e-5, max_steps=1000):
    Q = np.ones((env._ns(), env._na()))
    rewards = []
    states_seen = set()
    rl_params = {}
    rl_params['num_episodes'] = num_episodes
    rl_params['gamma'] = gamma
    rl_params['alpha'] = alpha
    rl_params['epsilon'] = epsilon
    rl_params['decay_rate'] = decay_rate
    rl_params['tolerance'] = tolerance
    rl_params['max_steps'] = max_steps
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
        rewards.append((cum_reward, iepisode))
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
    return Q, states_seen, rewards, rl_params

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
            indices_seen.add(curr_s)
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
    return Q, indices_seen

def back_out_policy(Q, indices_seen, env):
    indices_seen = sorted(list(indices_seen))
    actions = list(np.argmax(Q[indices_seen], axis=1))
    policy = {}
    for s, a in zip(indices_seen, actions):
        policy[env._i2s(s)] = a
    return policy

def nearest_neighbor_1d(s, states_seen):
    closest_distance = float('inf')
    closest_neighbor = None
    for state in states_seen:
        dist = abs(state[0] - s[0])
        if dist < closest_distance:
            closest_distance = dist
            closest_neighbor = state
    return closest_neighbor

def nearest_neighbor_2d(s, states_seen):
    closest_distance = float('inf')
    closest_neighbor = None
    for state in states_seen:
        dist = (state[0] - s[0]) ** 2 + (state[1] - s[1]) ** 2
        if (dist < closest_distance):
            closest_distance = dist
            closest_neighbor = state
    return closest_neighbor

def fill_policy(Q, indices_seen, env, policy):
    states_seen = convert_i2s(indices_seen, env)
    s_space_dims = env.s_space
    if len(s_space_dims) == 1:
        for i in s_space_dims[0]:
            if (i,) not in states_seen:
                neighbor = nearest_neighbor_1d((i,), states_seen)
                policy[(i,)] = policy[neighbor]
    elif len(s_space_dims) == 2:
        for i in s_space_dims[0]:
            for j in s_space_dims[1]:
                if (i, j) not in states_seen:
                    neighbor = nearest_neighbor_2d((i, j), states_seen)
                    policy[(i, j)] = policy[neighbor]
    return policy

def convert_i2s(indices_seen, env):
    states_seen = []
    for i in indices_seen:
        states_seen.append(env._i2s(i))
    return set(states_seen)

def render_single_episode(env, policy):
    episode_penalty = 0
    state = env._reset()
    done = False
    while not done:
        action = policy[state]
        state, penalty, done = env._step(action)
        episode_penalty += penalty
    assert done
    return episode_penalty

def evaluate_performance(env, num_trials=5, num_episodes=10):
    avg_penalty = []
    for trial_i in xrange(num_trials):
        Q, indices_seen, _, _ = q_learning(env)
        policy = back_out_policy(Q, indices_seen, env)
        policy = fill_policy(Q, indices_seen, env, policy)
        penalty_list = []
        for episode_i in xrange(num_episodes):
            episode_penalty = render_single_episode(env, policy)
            penalty_list.append(episode_penalty)
        avg_penalty.append(np.mean(penalty_list))
    return avg_penalty
