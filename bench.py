#!/usr/bin/env python3.9
import multiprocessing
import sys
import time
import timeit

def bench(size,nptype,number,repeat):
    durations = timeit.repeat(setup='import numpy as np; np.random.seed(42); a = [np.random.randint(low=-1000000,high=1000001,size='+str(size)+').astype(dtype=np.'+nptype+') for _ in range('+str(number)+')]',number=number,repeat=repeat,stmt='b,a = a[0],a[1:]; b.sort()',timer=time.process_time)
    print(min(durations)/number)

if __name__ == "__main__":
    sizes=[int(x) for x in sys.argv[1].split(",")]
    nptypes = sys.argv[2].split(",")
    numbers=[int(x) for x in sys.argv[3].split(",")]
    repeats=[int(x) for x in sys.argv[4].split(",")]
    for size in sizes:
        for nptype in nptypes:
            for number in numbers:
                for repeat in repeats:
                    print(size,nptype,number,repeat,end=' ')
                    p = multiprocessing.Process(target = bench, args = (size,nptype,number,repeat))
                    p.start()
                    p.join()