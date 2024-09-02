import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




RADIAL = False # True for radial field lines - False for dipole

RE = 6371.2

fig, ax = plt.subplots(figsize = (14, 7))

rl = 0.95 * RE
rh = 2.03 * RE
th = np.deg2rad(65)
RI = RE + 100
d = 10 # step used to illustrate radii just above and below ionosphere
C = np.sqrt(2 * (1 - np.cos(th))) # length of x axis
textsize = 14

x = np.linspace(-C/2, C/2, 101)
a = np.linspace(-th/2, th/2, 101)

cond = pd.read_csv('conductivity.csv')
cond.index = cond[cond.columns[0]]
cond = cond[['hall', 'pedersen']]

cond.index = cond.index + RE - 15
rs = np.linspace(rl, rh, 300)
cond_ = pd.DataFrame(index = rs, columns = cond.columns)
cond_new = pd.concat([cond, cond_]).sort_index().interpolate(limit = 10).fillna(0)

xx, yy = np.meshgrid(np.linspace(x[0] * RI, x[-1] * RI, 100), cond_new.index)
ss = np.interp(np.sqrt(xx**2 + yy**2).flatten(), cond_new.index, cond_new['pedersen']).reshape(xx.shape)

ccc = ax.contourf(xx, yy, ss, cmap = plt.cm.Blues, levels = np.linspace(0, 1.5e-4, 100), zorder = 0)








#ax.text(0, RI + 40, r'$\mathbf{B}^{+} = -\nabla V + \hat{\mathbf{r} \times \nabla T$', size = textsize, va = 'center')
#ax.text(0, RI + 20, r'$\mathbf{j}^{+} = \hat{\mathbf{r}} T$', size = textsize, va = 'center')
#ax.text(0, RI     , r'$\mathbf{j} = \hat{\mathbf{r}} \times (\mathbf{B}_h - \mathbf{B}^-_h)$', size = textsize, va = 'center', bbox=dict(facecolor='white', edgecolor='none', pad = 0))

#ax.text(RI * x[10], RI + 40, r'$\mathbf{v} = -\hat{\mathbf{r}}\times \nabla \Phi + \nabla_h W$', size = textsize)
#ax.text(RI * x[10], RI + 20, r'$\mathbf{E} = -\mathbf{v} \times \mathbf{B}$', size = textsize)

if RADIAL:
    for _ in a[::10]:
        ax.plot([rl * np.sin(_), 1.1*rh * np.sin(_)], [rl * np.cos(_), 1.1*rh * np.cos(_)], color = 'lightgrey', linewidth = .5, zorder = 0)
else:
    r_ = np.linspace(rl*0.9, rh*1.1, 100)
    for _ in a[::10]:
        req = RI / np.cos(np.pi/2 - _)**2
        ll = np.arccos(np.sqrt(r_ / req))
        xx, yy = r_ * np.cos(ll), r_ * np.sin(ll)
        ax.plot( xx, yy, color = 'lightgrey', linewidth = .5, zorder = 0)
        ax.plot(-xx, yy, color = 'lightgrey', linewidth = .5, zorder = 0)



ax.set_aspect('equal')

# Remove the ticks
ax.tick_params(
    axis='both',          # Apply to both x and y axes
    which='both',         # Apply to both major and minor ticks
    bottom=False,         # Turn off bottom ticks
    top=False,            # Turn off top ticks
    left=False,            # Turn off left ticks
    right=False,           # Turn off right ticks
    labelbottom=False,    # Turn off bottom tick labels
    labelleft=False       # Turn off left tick labels
)

# Set the spine colors to black
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

# increase width of frame
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)


ax.fill_between(x * RI, RE * np.cos(a), rl * np.cos(a), color = 'lightgrey')
ax.set_ylim(rl, rh)
ax.set_xlim(x[0] * RI, x[-1] * RI)



plt.tight_layout()

plt.savefig('equations_fig00.png', dpi = 250)




for cc in ccc.collections:
    cc.remove() 

ax.plot(RI * x, RI *  np.cos(a), linewidth = 5, color = 'C0')
ax.text((RI + d) * x[-5], (RI    ) *  np.cos(a[-5]), r'$R  $     ', rotation = -np.rad2deg(a[-5]), ha = 'left', va = 'center', size = textsize, color = 'C0', bbox=dict(facecolor='white', edgecolor='none', pad = 1))
#ax.text((RI + d) * x[-5], (RI + d) *  np.cos(a[-5]), r'$R^+$   ', rotation = -np.rad2deg(a[-5]), ha = 'left', va = 'center', size = textsize, color = 'C0', bbox=dict(facecolor='white', edgecolor='none', pad = 1))
#ax.text((RI + d) * x[-5], (RI - d) *  np.cos(a[-5]), r'$R^-$   ', rotation = -np.rad2deg(a[-5]), ha = 'left', va = 'center', size = textsize, color = 'C0', bbox=dict(facecolor='white', edgecolor='none', pad = 1))

plt.savefig('equations_fig01.png', dpi = 250)


ax.text(x[5] * (RE - 10), (RE - 10) * np.cos(a[5]), r'ground', rotation = -np.rad2deg(a[5]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
#ax.text(x[5] * (RE + 50), (RE + 50) * np.cos(a[5]), r'atmosphere:' + '\n' + r'$\mathbf{j} = 0$' + '\n' + r'$\mathbf{B} = -\nabla V_B$', rotation = -np.rad2deg(a[5]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
ax.text(x[5] * (RE + 50), (RE + 50) * np.cos(a[5]), r'atmosphere, $\mathbf{j} = 0$', rotation = -np.rad2deg(a[5]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
#ax.text(x[5] * (RI + 50), (RI + 50) * np.cos(a[5]), r'magnetosphere:' + '\n' + r'$\mathbf{j}_{\mathrm{FAC}} = \nabla\times\mathbf{B}/\mu_0$', rotation = -np.rad2deg(a[5]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
ax.text(x[5] * (RI + 50), (RI + 50) * np.cos(a[5]), r'"gap region" $\mathbf{j}\times\mathbf{B} = 0$', rotation = -np.rad2deg(a[5]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
ax.text(x[ 5] * (RI + 1 ), (RI + 1 ) * np.cos(a[ 5]), r'ionosphere', ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))


plt.savefig('equations_fig02.png', dpi = 250)

#ax.scatter((RI + d) * x[::2], (RI + d) *  np.cos(a[::2]), marker = '+', color = 'C0')
#ax.scatter((RI - d) * x[::2], (RI - d) *  np.cos(a[::2]), marker = '_', color = 'C0')


ax.text(x[15] * (RI + 1 ), (RI + 1 ) * np.cos(a[15]), r'$\partial B_r / \partial t = -\nabla \times \mathbf{E}$', rotation = -np.rad2deg(a[15]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))



#ax.text(x[5] * (RE - 10), (RE - 10) * np.cos(a[5]), r'ground', rotation = -np.rad2deg(a[5]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
#ax.text(x[5] * (RE + 50), (RE + 50) * np.cos(a[5]), r'atmosphere:' + '\n' + r'$\mathbf{j} = 0$' + '\n' + r'$\mathbf{B} = -\nabla V_B$', rotation = -np.rad2deg(a[5]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
#ax.text(x[5] * (RI + 50), (RI + 50) * np.cos(a[5]), r'magnetosphere:' + '\n' + r'$\mathbf{E} = -\mathbf{v} \times \mathbf{B}$' + '\n' + r'$\mathbf{j} = \nabla\times\mathbf{B}/\mu_0$', rotation = -np.rad2deg(a[5]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
plt.savefig('equations_fig03.png', dpi = 250)

#ax.text(x[ 5] * (RI + 1 ), (RI + 1 ) * np.cos(a[ 5]), r'ionosphere:' , rotation = -np.rad2deg(a[5]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
ax.text(x[40] * (RI + 1 ), (RI + 1 ) * np.cos(a[40]), r'$\mathbf{E} = (\Sigma_P\mathbf{j}\times\hat{\mathbf{r}} + \Sigma_H\mathbf{j})/(\Sigma_P^2 +\Sigma_H^2)$' , rotation = -np.rad2deg(a[40]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
ax.text(x[-35] * (RI + 1 ), (RI + 1 ) * np.cos(a[-35]), r'$\mathbf{j}_S = \hat{\mathbf{r}} \times (\mathbf{B}^+ - \mathbf{B}^-)/\mu_0$' , rotation = -np.rad2deg(a[-35]), ha = 'left', va = 'center', size = textsize, bbox=dict(facecolor='white', edgecolor='none', pad = 0))
plt.savefig('equations_fig04.png', dpi = 250)

#ax.text(x[-35] * (RI     ), (RI     ) * np.cos(a[-35]), r'$\mathbf{j}_S = \hat{\mathbf{r}} \times V_J - \nabla_S T_J$', size = textsize, va = 'center', bbox=dict(facecolor='white', edgecolor='none', pad = 0), rotation = -np.rad2deg(a[-35]))
#ax.text(x[-35] * (RI + 50), (RI + 50) * np.cos(a[-35]), r'$\mathbf{B} = -\nabla V_B + \hat{\mathbf{r}} \times \nabla T_B$' + '\n' + r'$\mathbf{v}_S = -\nabla W_v + \hat{\mathbf{r}}\times\nabla_S\Phi_V$'  + '\n' + r'$\mathbf{E}_S = -\nabla \Phi_E + \hat{\mathbf{r}}\times\nabla_SW_E$' + '\n' + r'$\mathbf{j} = T_{J_r}(R/r)^2\hat{\mathbf{r}}$' , size = textsize, va = 'center', rotation = -np.rad2deg(a[-35]))
#ax.text(x[-35] * (RI + 50), (RI + 20) * np.cos(a[-35]), r'$\mathbf{B} = -\nabla V_B + \hat{\mathbf{r}} \times \nabla T_B$', size = textsize, va = 'center', rotation = -np.rad2deg(a[-35]))
#plt.savefig('equations_fig3.png', dpi = 250)


plt.show()
plt.close()


