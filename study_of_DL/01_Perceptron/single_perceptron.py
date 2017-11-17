import numpy as np
from gate import gate

g = gate()

print("    OR        AND        NAND")
for xs in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    res1 = g.OR(xs[0], xs[1])
    res2 = g.AND(xs[0], xs[1])
    res3 = g.NAND(xs[0], xs[1])
    print(
        str(xs) + "->" + str(res1) + "  " + \
        str(xs) + "->" + str(res2) + "  " + \
        str(xs) + "->" + str(res3)
    )
