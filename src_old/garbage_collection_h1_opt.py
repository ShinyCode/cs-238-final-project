'''
Garbage Collection MDP v1 with Heap
State representation: (m_left, frag), with m_left = -1 as a sentinel for being out of memory
'''
import numpy as np
import allocator as ac
import sys
import garbage_collection_base as gcb

GC_PROB = 1

GC = 0
NGC = 1
CLS = 2

REWARD_GC = -10
REWARD_NGC = 0
REWARD_CLS = -5

REWARD_PASS = 0
REWARD_OOM = -500

STATE_OOM = -1

class GarbageCollectionEnvH1(gcb.GarbageCollectionEnv):
    def __init__(self, m_max, usage_pattern):
        gcb.GarbageCollectionEnv.__init__(self, m_max, usage_pattern, [GC, NGC, CLS], [range(-1, m_max + 1), range(0, ac.FRAG_STEPS + 1)])

    def _s0(self):
        return (self.m_max, self.alloc.fragmentation())

    def _name(self):
        return 'Garbage Collection with Fragmentation'

    def _next(self, s, a):
        sp = None
        ip = None
        r = 0
        if a == GC:
            self.alloc.do_gc()
            r += REWARD_GC
        elif a == NGC:
            r += REWARD_NGC
        elif a == CLS:
            r += REWARD_CLS
            self.alloc.do_cls()
        if self.alloc.do_allocate(self.usage_pattern[self.i]):
            ip = self.i + 1
            sp = (self.alloc.m_left(), self.alloc.fragmentation())
            r += REWARD_PASS
        else:
            ip = self.i
            sp = (STATE_OOM, self.alloc.fragmentation())
            r += REWARD_OOM
        return (sp, ip, r)
