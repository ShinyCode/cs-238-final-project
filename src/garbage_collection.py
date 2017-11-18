import numpy as np
import sys

GC = 0
NGC = 1

REWARD_PASS = 0
REWARD_GC = -10
REWARD_OOM = -500

class GarbageCollectionEnv(object):

    # m_max is total amount of memory available
    # usage_patern is a list of "malloc" amounts
    def __init__(self, m_max, usage_pattern):
        self.m_max = m_max
        self.usage_pattern = usage_pattern
        self._reset()
        self.a_space = set([GC, NGC])

    # return (sp, r, done)
    def _step(self, a):
        assert a in self.a_space
        sp, ip = self._next(self.s, a)
        r = self._reward(self.s, a)
        done = ip == len(self.usage_pattern)
        self.i = ip
        return (sp, r, done)

    def _seed(self, seed=None):
        pass

    def _next(self, s, a):
        sp = None
        ip = None
        if a == GC:
            sp = (self.m_max, self.m_max) # TODO: CHANGE THIS
            ip = self.i + 1
        elif a == NGC:
            if s[0] - self.usage_pattern[self.i] >= 0: # We have enough memory
                sp = (s[0] - self.usage_pattern[self.i], -self.usage_pattern[self.i]) # WANT TO ADVANCE self.i
                ip = self.i + 1
            else if s[0] >= 0: # We are about to run out of memory
                sp = (s[0] - self.usage_pattern[self.i], -self.usage_pattern[self.i]) # Don't want to advance self.i
                ip = self.i
            else: # We still don't have enough memory, i.e. s[0] < 0
                sp = s # Don't want to advance self.i
                ip = self.i
        return (sp, ip)


    def _reward(self, s, a):
        m_cur, m_change = s
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
        self.s = (m_max, 0)
        self.i = 0
