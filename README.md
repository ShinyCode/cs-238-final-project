# CS 238 Final Project

### Basic Usage

```python
import g_base
import g_ext
import usage_pattern as up
import mf

# g_base.GarbageCollectionEnv_Base OR g_ext.GarbageCollectionEnv_Ext
env = g_base.GarbageCollectionEnv_Base(m_max=100, usage_pattern=up.usage_pattern_flat(100, 5))
Q, indices_seen, rewards, rl_params = mf.q_learning(env)
policy = mf.back_out_policy(Q, indices_seen, env)
print policy
```
