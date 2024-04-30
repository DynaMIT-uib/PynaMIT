""" Construct constraint matrices and output resulting LaTeX code or Python code """

import sympy as sp
#sp.init_printing(use_unicode=True)

PRINT_LATEX = True # set to True to print LaTeX, False to print something more code-friendly

if PRINT_LATEX:
    br, bt, bp, ut, up, eP, eH, B0, Br = sp.symbols('b_r, b_theta, b_phi, u_theta, u_phi, eta_P, eta_H, B_0, B_r')
    d1r, d1t, d1p, d2r, d2t, d2p   = sp.symbols('d_1_r, d_1_theta, d_1_phi, d_2_r, d_2_theta, d_2_phi  ')
else:
    br, bt, bp, ut, up, eP, eH, B0, Br = sp.symbols('br, bt, bp, ut, up, eP, eH, B0, Br')
    d1r, d1t, d1p, d2r, d2t, d2p   = sp.symbols('d1r, d1t, d1p, d2r, d2t, d2p  ')

# Matrix that produces the horizontal electric field when muliplied to (J_theta, J_phi, 1)
A1 = sp.Matrix([[ eP * (bp**2 + br**2), -eP * bp * bt + eH * br, -up * Br],
                [-eP * bp * bt - eH * br, eP * (bt**2 + br**2),  ut * Br]])

# Adding a row which produces E_r
A2r = -sp.Matrix([[A1[0, 0] * bt/br + A1[1, 0] * bp/br , A1[0, 1] * bt/br + A1[1, 1] * bp/br, A1[0, 2] * bt/br + A1[1, 2] * bp/br]])
A2 =  sp.Matrix.vstack(A2r, A1)

if PRINT_LATEX:
    print('A2:')
    print(sp.latex(A2))

bBcrossE = sp.Matrix([[0, bp, -bt], [-bp, 0, br], [bt, -br, 0]]) / B0

# test that the cross product matrix works
x, y, z = sp.symbols('x y z')
xx = sp.Matrix([[x], [y], [z]])
bb = sp.Matrix([[br], [bt], [bp]]) / B0
print('this should be zeros: ', xx.cross(bb) - bBcrossE * xx, '\n\n')


A3 = bBcrossE * A2

if PRINT_LATEX:
    print('A3:')
    print(sp.latex(sp.simplify(A3)))
    print('\n\n')

dd = sp.Matrix([[d1r, d1t, d1p], [d2r, d2t, d2p]])
A4 = dd * A3
#print('A4:')
#print(sp.latex(A4))

A400 = sp.collect(sp.factor(A4[0, 0]), [eP, eH, ut, up]).expand()
A401 = sp.collect(sp.factor(A4[0, 1]), [eP, eH, ut, up]).expand()
A402 = sp.collect(sp.factor(A4[0, 2]), [eP, eH, ut, up]).expand()
A410 = sp.collect(sp.factor(A4[1, 0]), [eP, eH, ut, up]).expand()
A411 = sp.collect(sp.factor(A4[1, 1]), [eP, eH, ut, up]).expand()
A412 = sp.collect(sp.factor(A4[1, 2]), [eP, eH, ut, up]).expand()

A4_pars = []
for p in [eP, eH, ut, up]:
    A4_pars.append(sp.Matrix([[(A400.coeff(p)).simplify(), (A401.coeff(p)).simplify(), (A402.coeff(p)).simplify()],
                              [(A410.coeff(p)).simplify(), (A411.coeff(p)).simplify(), (A412.coeff(p)).simplify()]]))

if PRINT_LATEX:
    labels = ['{\\eta_P}', '{\\eta_H}', '{u_\\theta}', '{u_\\phi}']
    for i in range(4):
        print('\\mathbb A4_' + labels[i] + '=&')
        print(sp.latex(sp.simplify(A4_pars[i])))
        print('\\\\')
    print('\n\n')

    for i in range(2):
        print('\\mathbb A5_' + labels[i] + '=&')
        print(sp.latex(sp.simplify(A4_pars[i][:, :-1])))
        print('\\\\')
    print('\n\n')

    print('\\mathbf{c} = u_\\theta' + sp.latex(sp.simplify(A4_pars[2][:, -1])) + '+ u_\\phi' + sp.latex(sp.simplify(A4_pars[3][:, -1])))



# Write A5 and c, which are just A4 split up:
A5_eP, A5_eH = [A4_pars[0][:,: -1], A4_pars[1][:,: -1]]
c_ut, c_up   = [A4_pars[2][:,  -1], A4_pars[3][:,  -1]]

if not PRINT_LATEX: # print code:
    print('CODE:\n\n')
    print('# constant that is to be multiplied by u in theta direction:')
    print('c_ut_theta = ' + str(c_ut[0]))
    print('c_ut_phi   = ' + str(c_ut[1]))
    print('c_ut = np.hstack((c_ut_theta, c_ut_phi)).reshape((-1, 1)) # stack and convert to column vector\n')

    print('# constant that is to be multiplied by u in phi direction:')
    print('c_up_theta = ' + str(c_up[0]))
    print('c_up_phi   = ' + str(c_up[1]))
    print('c_up = np.hstack((c_up_theta, c_up_phi)).reshape((-1, 1)) # stack and convert to column vector\n\n')

    print('# A5eP matrix')
    print('a5eP00 = spr.diags(' + str(A5_eP[0, 0]) + ')') 
    print('a5eP01 = spr.diags(' + str(A5_eP[0, 1]) + ')') 
    print('a5eP10 = spr.diags(' + str(A5_eP[1, 0]) + ')') 
    print('a5eP11 = spr.diags(' + str(A5_eP[1, 1]) + ')')
    print('a5eP = spr.vstack((spr.hstack((a5eP00, a5eP01)), spr.hstack((a5eP10, a5eP11)))).tocsr()\n')

    print('# A5eH matrix')
    print('a5eH00 = spr.diags(' + str(A5_eH[0, 0]) + ')') 
    print('a5eH01 = spr.diags(' + str(A5_eH[0, 1]) + ')') 
    print('a5eH10 = spr.diags(' + str(A5_eH[1, 0]) + ')') 
    print('a5eH11 = spr.diags(' + str(A5_eH[1, 1]) + ')')
    print('a5eH = spr.vstack((sp.hstack((a5eH00, a5eH01)), spr.hstack((a5eH10, a5eH11)))).tocsr()')





