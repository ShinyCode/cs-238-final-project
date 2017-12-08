'''
Garbage Collection MDP v3
State representation: (m_left, o, dm_left)
    m_left - amount of memory left
    o - binary, whether we're out of memory
    dm_left - change in amount of memory from last time step
Example: (m_left=6, o=0, dm_left=?) --[M 16]--> (m_left=6, o=1, dm_left=0)
'''
import numpy as np
import sys

GC = 0
NGC = 1

REWARD_PASS = 0
REWARD_GC = -10
REWARD_OOM = -500

STATE_OOM = -1

class GarbageCollectionEnv3(object):
    # m_max is total amount of memory available
    # usage_patern is a list of "malloc" amounts
    def __init__(self, m_max, usage_pattern):
        self.m_max = m_max
        self.usage_pattern = usage_pattern
        self._reset(m_max, usage_pattern)
        self.a_space = set([GC, NGC])
        self._init_lookup()

    def _na(self):
        return len(self.a_space)

    def _ns(self):
        return (self.m_max + 1) * 2 * (2 * self.m_max + 1)

    # return (sp, r, done)
    def _step(self, a):
        assert a in self.a_space
        sp, ip = self._next(self.s, a)
        r = self._reward(self.s, a)
        done = ip == len(self.usage_pattern)
        self.i = ip
        self.s = sp
        return (sp, r, done)

    def _next(self, s, a):
        sp = None
        ip = None
        if a == GC:
            m_left = self.m_max - self.usage_pattern[self.i] # Memory left if we were to proceed
            if m_left >= 0: # We have enough memory to do the malloc
                sp = (m_left, False, m_left - s[0])
                ip = self.i + 1
            else:
                sp = (self.m_max, True, self.m_max - s[0]) # We garbage collected, but we still can't
                ip = self.i
        elif a == NGC:
            if s[0] - self.usage_pattern[self.i] >= 0: # We have enough memory
                sp = (s[0] - self.usage_pattern[self.i], False, -self.usage_pattern[self.i]) # WANT TO ADVANCE self.i
                ip = self.i + 1
            else: # We are about to run out of memory
                sp = (s[0], True, 0) # Don't want to advance self.i
                ip = self.i
        return (sp, ip)

    def _reward(self, s, a):
        if a == GC:
            return REWARD_GC
        if a == NGC:
            if s[0] - self.usage_pattern[self.i] >= 0: # We have enough memory
                return REWARD_PASS
            else: # Not enough memory
                return REWARD_OOM

    def _reset(self, m_max=None, usage_pattern=None):
        if m_max is not None:
            self.m_max = m_max
        if usage_pattern is not None:
            self.usage_pattern = usage_pattern
        self.s = (self.m_max, False, 0)
        self.i = 0
        return self.s

    # Given a state, outputs an index in the range [0, self._ns)
    def _init_lookup(self):
        s2i = {}
        i2s = {}
        i = 0
        for m_left in xrange(0, self.m_max + 1):
            for o in [False, True]:
                for dm_left in xrange(-self.m_max, self.m_max + 1):
                    s2i[(m_left, o, dm_left)] = i
                    i2s[i] = (m_left, o, dm_left)
                    i += 1
        assert i == self._ns()
        self.s2i = s2i
        self.i2s = i2s

    def _s2i(self, s):
        return self.s2i[s]

    def _i2s(self, i):
        return self.i2s[i]
