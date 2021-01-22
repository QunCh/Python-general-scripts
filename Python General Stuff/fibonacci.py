
from functools import reduce
import numpy as np
np.product(alist)
alist = range(1,11)
reduce(lambda x,y:x+y, map(lambda x: np.product(range(1,x+1)), alist))


def fib(n):
    if n==1:
        yield 0
    if n == 2:
        yield 1
    yield fib(n-1)+fib(n-2)
%time next(fib(30))
f = fib(50)
f
for num in f:
    print(num)

def fib1(n):
    alist = [0,1]
    for i in range(2,n+1):
        alist.append(alist[i-2]+alist[i-1])
    return alist

%timeit a = fib1(10000)

def fib2(n):
    a,b = 0,1
    for i in range(n):
        yield a
        a,b = b,a+b
%timeit a = list(fib2(10000))
