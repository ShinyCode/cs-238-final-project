
FRAG_STEPS = 20
GC_PROB = 1

ACTION_GC = 0
ACTION_NGC = 1
ACTION_CLS = 2

FREE = True
INUSE = False

STATE_OOM = -1

REWARD_GC = -10
REWARD_NGC = 0
REWARD_CLS = -5

REWARD_PASS = 0
REWARD_OOM = -500

def get_mdp_params():
    mdp_params = {}
    mdp_params['FRAG_STEPS'] = FRAG_STEPS
    mdp_params['GC_PROB'] = GC_PROB
    mdp_params['REWARD_GC'] = REWARD_GC
    mdp_params['REWARD_NGC'] = REWARD_NGC
    mdp_params['REWARD_CLS'] = REWARD_CLS
    mdp_params['REWARD_PASS'] = REWARD_PASS
    mdp_params['REWARD_OOM'] = REWARD_OOM
    return mdp_params
