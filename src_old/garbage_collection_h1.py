'''
Garbage Collection MDP v1 with Heap
State representation: (m_left, frag), with m_left = -1 as a sentinel for being out of memory
'''
import numpy as np
import allocator as ac
import sys

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

class GarbageCollectionEnv(object):
    # m_max is total amount of memory available
    # usage_patern is a list of "malloc" amounts
    def __init__(self, m_max, usage_pattern):
        print "Created garbage_collection_h1!"
        self.alloc = ac.Allocator(m_max, GC_PROB)
        self.m_max = m_max
        self.usage_pattern = usage_pattern
        self._reset(m_max, usage_pattern)
        self.a_space = set([GC, NGC, CLS])
        self._init_lookup()

    def _na(self):
        return len(self.a_space)

    def _ns(self):
        return (self.m_max + 2) * (ac.FRAG_STEPS + 1)

    # return (sp, r, done)
    def _step(self, a):
        assert a in self.a_space
        sp, ip, r = self._next(self.s, a)
        done = ip == len(self.usage_pattern)
        self.i = ip
        self.s = sp
        return (sp, r, done)

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

    def _reset(self, m_max=None, usage_pattern=None):
        if m_max is not None:
            self.m_max = m_max
        self.alloc.reset(self.m_max)
        if usage_pattern is not None:
            self.usage_pattern = usage_pattern
        self.s = (self.m_max, self.alloc.fragmentation())
        self.i = 0
        return self.s

    # Given a state, outputs an index in the range [0, self._ns)
    def _init_lookup(self):
        s2i = {}
        i2s = {}
        i = 0
        for m_left in xrange(-1, self.m_max + 1):
            for frag in xrange(0, ac.FRAG_STEPS + 1):
                s2i[(m_left, frag)] = i
                i2s[i] = (m_left, frag)
                i += 1
        assert i == self._ns()
        self.s2i = s2i
        self.i2s = i2s

    def _s2i(self, s):
        return self.s2i[s]

    def _i2s(self, i):
        return self.i2s[i]
