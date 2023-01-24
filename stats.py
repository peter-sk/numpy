#!/usr/bin/env python3
from sys import stdin, argv
from collections import defaultdict
from matplotlib import pyplot as plt
l2a2id = defaultdict(lambda: defaultdict(lambda: []))
for line in stdin:
    line = line.strip()
    if line:
        length, _, _, inputs, algorithm, duration = line.split()
        length, inputs, duration = int(length), int(inputs), float(duration)
        if not inputs & inputs-1:
            l2a2id[length][algorithm].append((inputs,duration))
for l, a2id in l2a2id.items():
    for a, ids in a2id.items():
        ids.sort()
n = 0
for l, a2id in l2a2id.items():
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
