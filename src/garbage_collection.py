'''
Garbage Collection MDP v1
State representation: (m_left), with m_left = -1 as a sentinel for being out of memory
'''
import numpy as np
import sys

GC = 0
NGC = 1

REWARD_PASS = 0
REWARD_GC = -10
REWARD_OOM = -500

STATE_OOM = -1

class GarbageCollectionEnv(object):
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
        return self.m_max + 2

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
            sp = self.m_max - self.usage_pattern[self.i]
            if sp >= 0: # We have enough memory to do the malloc
                ip = self.i + 1
            else:
                ip = self.i
                sp = STATE_OOM
        elif a == NGC:
            if s - self.usage_pattern[self.i] >= 0: # We have enough memory
                sp = s - self.usage_pattern[self.i] # WANT TO ADVANCE self.i
                ip = self.i + 1
            else: # We are about to run out of memory
                sp = STATE_OOM # Don't want to advance self.i
                ip = self.i
        return (sp, ip)

    def _reward(self, s, a):
        if a == GC:
            return REWARD_GC
        if a == NGC:
            if s - self.usage_pattern[self.i] >= 0: # We have enough memory
                return REWARD_PASS
            else: # Not enough memory
                return REWARD_OOM

    def _reset(self, m_max=None, usage_pattern=None):
        if m_max is not None:
            self.m_max = m_max
        if usage_pattern is not None:
            self.usage_pattern = usage_pattern
        self.s = self.m_max
        self.i = 0
        return self.s

    # Given a state, outputs an index in the range [0, self._ns)
    def _s2i(self, s):
        return s + 1

    def _i2s(self, i):
        return i - 1
