# CS 238 Final Project

### Basic Usage

```python
import garbage_collection as gc

env = gc.GarbageCollectionEnv(m_max=120, usage_pattern=[40, 60, 100, 40])
env._step(gc.NGC)
env._step(gc.GC)
```
