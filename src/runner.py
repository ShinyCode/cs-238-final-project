import garbage_collection as gc
import mf

def find_optimal_policy(env, agent):
    pass

def main():
    env = gc.GarbageCollectionEnv(120, [40, 0, 60, 0, 100, 0, 0, 0, 40, 0, 10, 10, 10])
    Q, states_seen = mf.q_learning(env)
    policy = mf.back_out_policy(Q, states_seen, env)
    print policy

if __name__ == '__main__':
    main()
