'''
Garbage Collection MDP v1 with Heap
State representation: (m_left, frag), with m_left = -1 as a sentinel for being out of memory
'''
import numpy as np
import allocator as ac
import sys
import garbage_collection as gcb
import constants as k

class GarbageCollectionEnv_H1(gcb.GarbageCollectionEnv):
    def __init__(self, m_max, usage_pattern):
        gcb.GarbageCollectionEnv.__init__(self, m_max, usage_pattern, [k.ACTION_GC, k.ACTION_NGC, k.ACTION_CLS], [range(-1, m_max + 1), range(0, k.FRAG_STEPS + 1)])

    def _s0(self):
        return (self.m_max, self.alloc.fragmentation())

    def _name(self):
        return 'Garbage Collection with Fragmentation'

    def _next(self, s, a):
        sp = None
        ip = None
        r = 0
        if a == k.ACTION_GC:
            self.alloc.do_gc()
            r += k.REWARD_GC
        elif a == k.ACTION_NGC:
            r += k.REWARD_NGC
        elif a == k.ACTION_CLS:
            r += k.REWARD_CLS
            self.alloc.do_cls()
        if self.alloc.do_allocate(self.usage_pattern[self.i]):
            ip = self.i + 1
            sp = (self.alloc.m_left(), self.alloc.fragmentation())
            r += k.REWARD_PASS
        else:
            ip = self.i
            sp = (k.STATE_OOM, self.alloc.fragmentation())
            r += k.REWARD_OOM
        return (sp, ip, r)
