import numpy as np

def usage_pattern_flat(length, level):
    return [level] * length

def usage_pattern_square(length, min_level, min_length, max_level, max_length):
    pattern = []
    for i in xrange(length):
        if i % (min_length + max_length) < min_length:
            pattern.append(min_level)
        else:
            pattern.append(max_level)
    return pattern

def usage_pattern_rising(length, min_level, max_level):
    pattern = []
    for i in xrange(length):
        pattern.append(int(min_level + (i / float(length)) * (max_level - min_level)))
    return pattern

def tile_pattern(pattern, length): # Total length of the final pattern
    num_reps = length / len(pattern)
    tiled_pattern = pattern * (num_reps + 1)
    return tiled_pattern[:length]

def usage_pattern_sawtooth(length, pulse_length, min_level, max_level):
    return tile_pattern(usage_pattern_rising(pulse_length, min_level, max_level), length)

def usage_pattern_random(length, min_level, max_level):
    return [np.random.randint(min_level, max_level + 1) for x in range(length)]
