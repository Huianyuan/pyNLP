import numpy as np

A = np.random.randn(4, 3)
B = np.sum(A, axis=1, keepdims=True)
C = np.sum(A, axis=1)
print(A)
print("------------")
print(B)
print("------------")
print(C)