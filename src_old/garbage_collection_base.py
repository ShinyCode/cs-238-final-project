'''
Garbage Collection MDP Base
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
    # env = GarbageCollectionEnv(m_max, usage_pattern, [a0, a1,...], [[s_a0, s_a1,...], [s_b0, s_b1,...],...])
    def __init__(self, m_max, usage_pattern, a_space, s_space):
        print self._name()
        self.alloc = ac.Allocator(m_max, GC_PROB)
        self.m_max = m_max
        self.usage_pattern = usage_pattern
        self._reset(m_max, usage_pattern)
        self.a_space = set(a_space)
        self.s_space = s_space
        self._init_lookup()

    def _na(self):
        return len(self.a_space)

    def _ns(self):
        ns = 1
        for dim in self.s_space:
            ns *= len(dim)
        return ns

    def _s0(self):
        raise NotImplementedError

    def _name(self):
        raise NotImplementedError
        return 'no name!'

    # return (sp, r, done)
    def _step(self, a):
        assert a in self.a_space
        sp, ip, r = self._next(self.s, a)
        done = ip == len(self.usage_pattern)
        self.i = ip
        self.s = sp
        return (sp, r, done)

    def _next(self, s, a):
        raise NotImplementedError
        sp = None
        ip = None
        r = 0
        return (sp, ip, r)

    def _reset(self, m_max=None, usage_pattern=None):
        if m_max is not None:
            self.m_max = m_max
        self.alloc.reset(self.m_max)
        if usage_pattern is not None:
            self.usage_pattern = usage_pattern
        self.s = self._s0()
        self.i = 0
        return self.s

    # Given a state, outputs an index in the range [0, self._ns)
    def _init_lookup(self):
        s2i = {}
        i2s = {}
        i_wrap = [0]
        self._enum_states(0, i_wrap, [], s2i, i2s)
        assert i_wrap[0] == self._ns()
        self.s2i = s2i
        self.i2s = i2s

    def _enum_states(self, idim, i_wrap, s, s2i, i2s):
        if idim == len(self.s_space):
            s_tuple = tuple(s)
            s2i[s_tuple] = i_wrap[0]
            i2s[i_wrap[0]] = s_tuple
            i_wrap[0] += 1
            return
        for val in self.s_space[idim]:
            s.append(val)
            self._enum_states(idim + 1, i_wrap, s, s2i, i2s)
            del s[-1]

    def _s2i(self, s):
        return self.s2i[s]

    def _i2s(self, i):
        return self.i2s[i]
