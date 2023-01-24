#!/usr/bin/env python3.9
from sys import stdin, argv, path
del path[0]
from collections import defaultdict
from matplotlib import pyplot as plt
l2a2id = defaultdict(lambda: defaultdict(lambda: []))
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
        l2a2id[length][ialgorithm].append((iinputs,duration))
for l, a2id in l2a2id.items():
    for a, ids in a2id.items():
        ids.sort()
n = 0
for l, a2id in l2a2id.items():
    print(n)
    n += 1
    vals = []
    #plt.figure(n)
    plt.subplot(2,3,n)
    plt.title("list length = %d" % l)
    plt.xscale("log", base=2)
    plt.xticks([2**n for n in range(0,9)])
    for a, ids in sorted(a2id.items()):
        if a in argv[1:]:
            continue
        i, d = zip(*ids)
        plt.plot(i, d, label=a)
        vals.extend(d)
    plt.legend()
    plt.ylim(0,sorted(vals, reverse=True)[2]*1.1)
plt.show()
