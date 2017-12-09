'''
Garbage Collection MDP v2
State representation: (m_left, o)
    m_left - amount of memory left
    o - binary, whether we're out of memory
Example: (m_left=6, o=0) --[M 16]--> (m_left=6, o=1)
'''
import numpy as np
import sys

GC = 0
NGC = 1

REWARD_PASS = 0
REWARD_GC = -10
REWARD_OOM = -500

STATE_OOM = -1

class GarbageCollectionEnv2(object):
    # m_max is total amount of memory available
    # usage_patern is a list of "malloc" amounts
    def __init__(self, m_max, usage_pattern):
        self.m_max = m_max
        self.usage_pattern = usage_pattern
        self._reset(m_max, usage_pattern)
        self.a_space = set([GC, NGC])

    def _na(self):
        return len(self.a_space)

    def _ns(self):
        return (self.m_max + 1) * 2

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
                sp = (m_left, False)
                ip = self.i + 1
            else:
                sp = (self.m_max, True) # We garbage collected, but we still can't
                ip = self.i
        elif a == NGC:
            if s[0] - self.usage_pattern[self.i] >= 0: # We have enough memory
                sp = (s[0] - self.usage_pattern[self.i], False) # WANT TO ADVANCE self.i
                ip = self.i + 1
            else: # We are about to run out of memory
                sp = (s[0], True) # Don't want to advance self.i
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
        self.s = (self.m_max, False)
        self.i = 0
        return self.s

    # Given a state, outputs an index in the range [0, self._ns)
    def _s2i(self, s):
        m_left, oom = s # m_left in [0, m_max]
        oom = 1 if oom else 0 # oom in [0, 1]
        return oom * (self.m_max + 1) + m_left

    def _i2s(self, i):
        return (i % (self.m_max + 1), i / (self.m_max + 1))
