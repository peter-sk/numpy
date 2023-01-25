#!/usr/bin/env python3.9
from sys import stdin, argv, path
del path[0]
from collections import defaultdict
from itertools import cycle
from matplotlib import pyplot as plt

style = cycle(["-","--","-.",":",".","h","H"])

l2d2a2id = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
for line in stdin:
    line = line.strip()
    if line and line[0] != '#':
        try:
            length, dtype, _, _, ialgorithm, iinputs, einputs, ealgorithm, duration = line.split()
        except ValueError as e:
            print(line)
            raise e
        length, iinputs, einputs, duration = int(length), int(iinputs), int(einputs), float(duration)
#        if not inputs & inputs-1:
#        if ialgorithm == ealgorithm:
        l2d2a2id[length][dtype][ialgorithm].append((iinputs,duration))
for l, d2a2id in l2d2a2id.items():
    for d, a2id in d2a2id.items():
        for a, ids in a2id.items():
            ids.sort()

n = 0
for l, d2a2id in l2d2a2id.items():
    n += 1
    plt.figure(n)
    m = 0
    for d, a2id in d2a2id.items():
        m += 1
        vals = []
        plt.subplot(2,4,m)
        style = cycle(['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted'])
        plt.title("list length = %d, dtype = %s" % (l, d))
        plt.xscale("log", base=2)
        plt.xticks([2**n for n in range(0,9)])
        for a, ids in sorted(a2id.items()):
            if argv[1:] and a not in argv[1:]:
                continue
            i, d = zip(*ids)
            plt.plot(i, d, label=a, linestyle=next(style))
            vals.extend(d)
        plt.legend()
        plt.ylim(0,sorted(vals, reverse=True)[2]*1.1)
plt.show()
