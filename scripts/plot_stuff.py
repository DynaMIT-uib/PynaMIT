# visualization

import pynamit
import numpy as np
import matplotlib.pyplot as plt

ts = np.linspace(0,120, 5)

#ts = np.linspace(1,240, 40)
path = '/Users/laundal/Dropbox/git/dynamit/PynaMIT/scripts/aurora_update'

a = pynamit.PynamEye(path)

for t in ts:
    a.set_time(t)
    a.make_multipanel_output_figure()
    
    plt.savefig('fig{:.2f}.png'.format(t))


print(3/0)
m_ss = np.load('/Users/laundal/Dropbox/git/dynamit/PynaMIT/notebooks/mvss.npy')
a.m_ind = m_ss
a.derive_E_from_B()
#a.derive_E_from_B()

a.m_Phi = a.RI * a.m_Phi
a.m_W = a.RI * a.m_W
a.make_multipanel_output_figure()

plt.savefig('fig_ss.png')



