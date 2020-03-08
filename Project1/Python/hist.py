import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Where to save the figures and data files
DATA_ID = "Results"

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

X = pd.read_fwf(data_path("Onebody_Density.dat"))



print(X)
X["Counter"]/=1e6
counts = []
bins = []

y = np.zeros(len(X["Counter"]))
r = 0.01
for i in range(len(X["Counter"])):
    V = 4*(i*(i+1)+ 1/3)*np.pi*r**3
    y[i] = int(X["Counter"][i]/V)

x = np.linspace(0, 2, 200)

import seaborn as sns
sns.set()
plt.plot(x, X["Counter"])
plt.show()
#
