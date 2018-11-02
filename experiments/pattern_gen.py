import copy

size = 3

pattern_len = size ** 2
elems = [-1, 0, 1]


def check_pattern(pattern):
    pass


def generate_pattern(elems, pattern_len, pattern=None):
    if pattern is not None and len(pattern) == pattern_len:
        print(pattern)
        check_pattern(pattern)
        return pattern
    for elem in elems:
        if pattern is None:
            pattern = []
        _pattern = copy.deepcopy(pattern)
        _pattern.append(elem)
        generate_pattern(elems, pattern_len, _pattern)


generate_pattern(elems, pattern_len)
