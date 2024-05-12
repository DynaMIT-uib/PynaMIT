""" Construct constraint matrices and output resulting LaTeX code or Python code """

import sympy as sp
#sp.init_printing(use_unicode=True)

PRINT_LATEX = True # set to True to print LaTeX, False to print something more code-friendly

if PRINT_LATEX:
    br, bt, bp, ut, up, eP, eH, B0, Br = sp.symbols('b_r, b_theta, b_phi, u_theta, u_phi, eta_P, eta_H, B_0, B_r')
    e1r, e1t, e1p, e2r, e2t, e2p   = sp.symbols('e_1_r, e_1_theta, e_1_phi, e_2_r, e_2_theta, e_2_phi  ')
else:
    br, bt, bp, ut, up, eP, eH, B0, Br = sp.symbols('br, bt, bp, ut, up, eP, eH, B0, Br')
    e1r, e1t, e1p, e2r, e2t, e2p   = sp.symbols('e1r, e1t, e1p, e2r, e2t, e2p  ')

# Matrix that produces the horizontal electric field when muliplied to (J_theta, J_phi, 1)
A2D_ = sp.Matrix([[ eP * (bp**2 + br**2), -eP * bp * bt + eH * br, -up * Br],
                  [-eP * bp * bt - eH * br, eP * (bt**2 + br**2),  ut * Br]])

# Adding a row which produces E_r
Ar   = -sp.Matrix([[A2D_[0, 0] * bt/br + A2D_[1, 0] * bp/br , A2D_[0, 1] * bt/br + A2D_[1, 1] * bp/br, A2D_[0, 2] * bt/br + A2D_[1, 2] * bp/br]])
A3D_ =  sp.Matrix.vstack(Ar, A2D_)

#if PRINT_LATEX:
#    print('E = X (Jth, Jph, 1):')
#    print(sp.latex(A3D_))

# A3D_, multiplied by any vector [x, y, 1] should give a vector that is perpendicular to b. Check:
assert (A3D_ * sp.Matrix([[sp.symbols('x')], [sp.symbols('y')], [1]])).dot(sp.Matrix([br, bt, bp])).simplify() == 0


ee    = sp.Matrix([[e1r, e1t, e1p], [e2r, e2t, e2p]])
A_Eei = ee * A3D_
alpha11 = A_Eei[0, 0].simplify()
alpha12 = A_Eei[0, 1].simplify()
alpha13 = A_Eei[0, 2].simplify()
alpha21 = A_Eei[1, 0].simplify()
alpha22 = A_Eei[1, 1].simplify()
alpha23 = A_Eei[1, 2].simplify()

alpha11 = sp.collect(sp.factor(alpha11), [eP, eH]).expand()
alpha12 = sp.collect(sp.factor(alpha12), [eP, eH]).expand()
alpha13 = sp.collect(sp.factor(alpha13), [eP, eH]).expand()
alpha21 = sp.collect(sp.factor(alpha21), [eP, eH]).expand()
alpha22 = sp.collect(sp.factor(alpha22), [ut, up]).expand()
alpha23 = sp.collect(sp.factor(alpha23), [ut, up]).expand()

if PRINT_LATEX:

    print('\\alpha_{11}=&' + '\\eta_P('+ sp.latex(alpha11.coeff(eP))   + ')    && + \\eta_H('+ sp.latex(alpha11.coeff(eH)) + ')\\nonumber \\\\')
    print('=&                 \\eta_P\\alpha_{11}^{\\eta_P}                    && + \\eta_H\\alpha_{11}^{\\eta_H}                         \\\\')
    print('\\alpha_{12}=&' + '\\eta_P('+ sp.latex(alpha12.coeff(eP))   + ')    && + \\eta_H('+ sp.latex(alpha12.coeff(eH)) + ')\\nonumber \\\\')
    print('=&                 \\eta_P\\alpha_{12}^{\\eta_P}                    && + \\eta_H\\alpha_{12}^{\\eta_H}                         \\\\')
    print('\\alpha_{21}=&' + '\\eta_P('+ sp.latex(alpha21.coeff(eP))   + ')    && + \\eta_H('+ sp.latex(alpha21.coeff(eH)) + ')\\nonumber \\\\')
    print('=&                 \\eta_P\\alpha_{21}^{\\eta_P}                    && + \\eta_H\\alpha_{21}^{\\eta_H}                         \\\\')
    print('\\alpha_{22}=&' + '\\eta_P('+ sp.latex(alpha22.coeff(eP))   + ')    && + \\eta_H('+ sp.latex(alpha22.coeff(eH)) + ')\\nonumber \\\\')
    print('=&                 \\eta_P\\alpha_{22}^{\\eta_P}                    && + \\eta_H\\alpha_{22}^{\\eta_H}                         \\\\')
    print('\\alpha_{13}=&' + 'u_\\theta('+ sp.latex(alpha23.coeff(ut)) + ')    && + u_\\phi('+ sp.latex(alpha23.coeff(up)) + ')\\nonumber \\\\')
    print('=&                 u_\\theta\\alpha_{13}^{u_\\theta}                && + u_\\phi\\alpha_{13}^{u_\\phi}                         \\\\')
    print('\\alpha_{23}=&' + 'u_\\theta('+ sp.latex(alpha23.coeff(ut)) + ')   && + u_\\phi('+ sp.latex(alpha23.coeff(up)) + ')\\nonumber \\\\')
    print('=&                 u_\\theta\\alpha_{23}^{u_\\theta}                && + u_\\phi\\alpha_{23}^{u_\\phi}                             ')


print(3/0)






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
    print('c_ut_theta = ' + str(sp.simplify(c_ut[0])))
    print('c_ut_phi   = ' + str(sp.simplify(c_ut[1])))
    print('c_ut = np.hstack((c_ut_theta, c_ut_phi)).reshape((-1, 1)) # stack and convert to column vector\n')

    print('# constant that is to be multiplied by u in phi direction:')
    print('c_up_theta = ' + str(sp.simplify(c_up[0])))
    print('c_up_phi   = ' + str(sp.simplify(c_up[1])))
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
    print('a5eH = spr.vstack((spr.hstack((a5eH00, a5eH01)), spr.hstack((a5eH10, a5eH11)))).tocsr()')





