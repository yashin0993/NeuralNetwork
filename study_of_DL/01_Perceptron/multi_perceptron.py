import numpy as np
from gate import gate

g = gate()

print("XOR")
for xs in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    res = g.XOR(xs[0], xs[1])
    print(str(xs) + "->" + str(res))