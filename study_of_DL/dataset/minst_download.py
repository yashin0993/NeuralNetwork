import os, sys
import numpy
from mnist import load_mnist

load_mnist(normalize=True, flatten=True, one_hot_label=False)