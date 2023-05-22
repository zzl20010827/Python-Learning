import numpy as np

def f(a,b):
    a[0]=5
    b[0]=1
    c=1
    
a=np.array([1,2,3])
b=np.array([0])
c=0
print(a,b,c)
f(a,b)
print(a,b,c)