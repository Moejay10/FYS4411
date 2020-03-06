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



import seaborn as sns
sns.distplot(X["Counter"])
plt.show()
#
