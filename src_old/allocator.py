import numpy as np
import sys

FRAG_STEPS = 20

GC = 0
NGC = 1
CLS = 2

FREE = True
INUSE = False

class Allocator(object):

    def __init__(self, m_max, gc_prob):
        self.heap = [(m_max, FREE)]
        self.m_max = m_max
        self.gc_prob = gc_prob

    def reset(self, m_max=None):
        if m_max is None:
            m_max = self.m_max
        self.m_max = m_max
        self.heap = [(m_max, FREE)]

    def do_allocate(self, m): # Try to allocate m amount of memory, returns whether it was successful
        for i, block in enumerate(self.heap):
            if block[1] != FREE or block[0] < m: # Either not free, or not big enough
                continue
            old_block = self.heap[i]
            self.heap[i] = (old_block[0] - m, FREE)
            self.heap.insert(i, (m, INUSE))
            return True
        return False

    def m_left(self):
        return sum([block[0] for block in self.heap if block[1] == FREE])

    def fragmentation(self):
        free_blocks = [block[0] for block in self.heap if block[1] == FREE]
        if sum(free_blocks) == 0:
            return 0
        return int((1.0 - float(max(free_blocks)) / sum(free_blocks)) * FRAG_STEPS)

    def do_gc(self):
        num_blocks_freed = 0
        m_freed = 0
        for i, block in enumerate(self.heap):
            if block[1] == FREE:
                continue
            if np.random.uniform() < self.gc_prob:
                self.heap[i] = (block[0], FREE)
                num_blocks_freed += 1
                m_freed += block[0]
        return (num_blocks_freed, m_freed)

    def do_cls(self):
        i_start = None
        for i, block in enumerate(self.heap):
            if block[1] == INUSE:
                i_start = None
                continue
            # The block is free
            if i_start is None:
                i_start = i
            else:
                self.heap[i_start] = (self.heap[i_start][0] + block[0], FREE)
                self.heap[i] = None
        self.heap = filter(lambda x: x is not None, self.heap)
