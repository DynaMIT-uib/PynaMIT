# visualization

import pynamit
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/laundal/Dropbox/git/dynamit/PynaMIT/scripts/hdtest'
a = pynamit.PynamEye(path).set_time(212)




a.make_multipanel_output_figure()
plt.savefig('fig_212.png')



m_ss = np.load('/Users/laundal/Dropbox/git/dynamit/PynaMIT/notebooks/mvss.npy')
a.m_ind = m_ss
a.derive_E_from_B()
#a.derive_E_from_B()

a.make_multipanel_output_figure()

plt.savefig('fig_ss.png')



