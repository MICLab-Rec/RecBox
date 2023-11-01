import numpy as np
import torch
import os
import pandas as pd
from random import random
from time import sleep
from tqdm import tqdm

from matplotlib import pyplot as plt

plt.plot(np.arange(start = -3,stop = 3,step = 0.1),torch.sinh(torch.FloatTensor(np.arange(start = -3,stop = 3,step = 0.1))))
plt.show()
#sss
