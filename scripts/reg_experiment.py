import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


gtg  = np.load("gtg.npy")
gtd  = np.load("gtd.npy")
n    = np.load("_n.npy")
PFAC = np.load("pfac.npy")
G    = np.load('_G.npy')
d    = np.load('_d.npy')



I = np.eye(n.size)
R = I * n * (n + 1) / (2 * n + 1) + PFAC * (n + 1)

m_plain = np.linalg.lstsq(gtg, gtd, rcond = 0)[0]
alpha = np.logspace(-2, 4, 20)
ms = []
for a in alpha:
    ms.append(np.linalg.lstsq(gtg + R * a, gtd, rcond = 0)[0])

misfits = [np.sqrt(np.mean((G.dot(m) - d)**2)) for m in ms]
norms   = [np.linalg.norm(m) for m in ms]

fig, ax = plt.subplots(ncols = 2)
ax[0].plot(misfits, norms, 'o-')
ax[0].scatter(misfits[-5], norms[-5], zorder = 5, c = 'red')
ax[0].set_xlabel('misfit')
ax[0].set_ylabel('model_norm')

df_plain = pd.DataFrame({'n':n, 'coeff':m_plain, 'pcoeff':PFAC.dot(m_plain)})
df_reg   = pd.DataFrame({'n':n, 'coeff':ms[-5] , 'pcoeff':PFAC.dot(ms[-5] )})


nn = np.unique(n)
p_plain = (df_plain.coeff**2).groupby(df_plain.n).sum() * nn * (nn + 1.) / (2 * nn + 1.) + (df_plain.pcoeff**2).groupby(df_plain.n).sum() * ( nn + 1)
p_reg   = (df_reg  .coeff**2).groupby(df_reg  .n).sum() * nn * (nn + 1.) / (2 * nn + 1.) + (df_reg  .pcoeff**2).groupby(df_reg  .n).sum() * ( nn + 1)






plt.show()