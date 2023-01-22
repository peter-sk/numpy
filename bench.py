#!/usr/bin/env python3.9
import datetime
import multiprocessing
import pathlib
import subprocess
import sys
import time
import timeit

def bench(size,nptype,number,repeat):
    sys.path[0] = str(next(pathlib.Path("build").glob('lib*')))
    durations = timeit.repeat(setup='import numpy as np; np.random.seed(42); a = [np.random.randint(low=-1000000,high=1000001,size='+str(size)+').astype(dtype=np.'+nptype+') for _ in range('+str(number)+')]',number=number,repeat=repeat,stmt='b,a = a[0],a[1:]; b.sort()',timer=time.process_time)
    print(min(durations)/number)

def power(n):
    i = 1
    while i < n:
        i *= 2
    return i

def compile(sorttype,inputs):
    main, sub = sorttype.split('_')
    print("# compiling %s_%s for %d inputs at %s" % (main, sub, inputs, str(datetime.datetime.now())))
    main, sub = main.upper(), sub.upper()
    npysort = pathlib.Path("numpy") / "core" / "src" / "npysort"
    [p.touch() for p in npysort.glob("*q*.cpp")]
    config = npysort / "npysort_config.h"
    text = config.read_text().split("//")[0]+"""//
#define NPY_SORT_TYPE NPY_SORT_%s
#define %s_TYPE %s_%s
#define NPY_SORT_BASE %d
#define NPY_SORT_POWER %d
""" % (main,main,main,sub,inputs,power(inputs))
    config.write_text(text)
    try:
        subprocess.run([sys.executable,'setup.py','build','-j','8'],check=True,capture_output=True)
    except subprocess.CalledProcessError as e:
        print("STDOUT:\n%s\nSTDERR:\n%s" % (e.stdout,e.stderr))
        raise e

if __name__ == "__main__":
    args = sys.argv[1:]
    sorttypes=args[0].split(",")
    inputss=[int(x) for x in args[1].split(",")]
    sizes=[int(x) for x in args[2].split(",")]
    nptypes = args[3].split(",")
    numbers=[int(x) for x in args[4].split(",")]
    repeats=[int(x) for x in args[5].split(",")]
    for sorttype in sorttypes:
        for inputs in inputss:
            compile(sorttype,inputs)
            for size in sizes:
                for nptype in nptypes:
                    for number in numbers:
                        for repeat in repeats:
                            print(size,nptype,number,repeat,sorttype,inputs,end=' ')
                            p = multiprocessing.Process(target = bench, args = (size,nptype,number,repeat))
                            p.start()
                            p.join()