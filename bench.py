#!/usr/bin/env python3
import sys
import time
import timeit
size=int(sys.argv[1])
nptype=sys.argv[2]
number=int(sys.argv[3])
repeat=int(sys.argv[4])
#duration = timeit.repeat(setup='import numpy as np; a = [np.random.randint(low=-1000000,high=1000001,size='+str(size)+').astype(dtype=np.'+nptype+') for _ in range('+str(number)+')]',number=number,repeat=repeat,stmt='b,a = a[0],a[1:]; b.sort()',timer=time.process_time)
duration = timeit.repeat(setup='import numpy as np; np.random.seed(42); a = [np.random.randint(low=-1000000,high=1000001,size='+str(size)+').astype(dtype=np.'+nptype+') for _ in range('+str(number)+')]',number=number,repeat=repeat,stmt='b,a = a[0],a[1:]; b.sort()',timer=time.process_time)
print(min(duration))
