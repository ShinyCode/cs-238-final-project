import garbage_collection_h1_opt as gc
import mf
import usage_pattern as up

def find_optimal_policy(env, agent):
    pass

def main():
    # env = gc.GarbageCollectionEnv(120, [40, 0, 60, 0, 100, 0, 0, 0, 40, 0, 10, 10, 10])
    # env = gc.GarbageCollectionEnv(100, up.usage_pattern_square(100, 0, 5, 10, 5))
    env = gc.GarbageCollectionEnvH1(100, up.usage_pattern_random(100, 0, 10))
    Q, states_seen = mf.q_learning(env, num_episodes=5000, max_steps=1000, epsilon=0.9999)
    #Q, states_seen = mf.sarsa(env)
    policy = mf.back_out_policy(Q, states_seen, env)
    print policy

if __name__ == '__main__':
    main()
