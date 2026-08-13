import numpy as np
import matplotlib.pyplot as plt
fn = 'curie_results.txt'
data = np.loadtxt(fn)
T = data[:,0]
M = data[:,1]
plt.scatter(T,M)
plt.show()
