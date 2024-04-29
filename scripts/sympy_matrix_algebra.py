import sympy as sp
#sp.init_printing(use_unicode=True)

br, bt, bp, ut, up, eP, eH, B0, Br = sp.symbols('b_r, b_theta, b_phi, u_theta, u_phi, eta_P, eta_H, B_0, B_r')
d1r, d1t, d1p, d2r, d2t, d2p   = sp.symbols('d_1_r, d_1_theta, d_1_phi, d_2_r, d_2_theta, d_2_phi  ')
#br, bt, bp, ut, up, eP, eH, B0, Br = sp.symbols('br, bt, bp, ut, up, eP, eH, B0, Br')
#d1r, d1t, d1p, d2r, d2t, d2p   = sp.symbols('d1r, d1t, d1p, d2r, d2t, d2p  ')

# Matrix that produces the horizontal electric field when muliplied to (J_theta, J_phi, 1)
A1 = sp.Matrix([[ eP * (bp**2 + br**2), -eP * bp * bt + eH * br, -up * Br],
                [-eP * bp * bt - eH * br, eP * (bt**2 + br**2),  ut * Br]])

# Adding a row which produces E_r
A2r = -sp.Matrix([[A1[0, 0] * bt/br + A1[1, 0] * bp/br , A1[0, 1] * bt/br + A1[1, 1] * bp/br, A1[0, 2] * bt/br + A1[1, 2] * bp/br]])
A2 =  sp.Matrix.vstack(A2r, A1)

print('A2:')
print(sp.latex(A2))

bBcrossE = sp.Matrix([[0, bp, -bt], [-bp, 0, br], [bt, -bp, 0]]) / B0
A3 = bBcrossE * A2
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

for i in range(4):
    print('A4_' + str(i + 1) + '=&')
    print(sp.latex(sp.simplify(A4_pars[i])))
    print('\\\\')

for i in range(4):
    print('A4_' + str(i + 1) + '=&')
    print(sp.simplify(A4_pars[i]))


# Write A5 and c, which are just A4 split up:
A5_pars = []
c_pars = []




