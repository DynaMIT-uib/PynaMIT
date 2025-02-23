"""Output LaTeX code or Python code of constraint matrices."""

import sympy as sp

# sp.init_printing(use_unicode=True)
# True for LaTeX print, False for more code-friendly print.
PRINT_LATEX = False

if PRINT_LATEX:
    br, bt, bp, ut, up, eP, eH, B0, Br = sp.symbols(
        "b_r, b_theta, b_phi, u_theta, u_phi, eta_P, eta_H, B_0, B_r"
    )
    e1r, e1t, e1p, e2r, e2t, e2p = sp.symbols(
        "e_1_r, e_1_theta, e_1_phi, e_2_r, e_2_theta, e_2_phi  "
    )
else:
    br, bt, bp, ut, up, eP, eH, B0, Br = sp.symbols(
        "br, bt, bp, ut, up, eP, eH, B0, Br"
    )
    e1r, e1t, e1p, e2r, e2t, e2p = sp.symbols("e1r, e1t, e1p, e2r, e2t, e2p  ")

# Construct matrix producing horizontal electric field when muliplied
# with (J_theta, J_phi, 1)
A2D_ = sp.Matrix(
    [
        [eP * (bp**2 + br**2), -eP * bp * bt + eH * br, -up * Br],
        [-eP * bp * bt - eH * br, eP * (bt**2 + br**2), ut * Br],
    ]
)

# Add row which produces E_r.
Ar = -sp.Matrix(
    [
        [
            A2D_[0, 0] * bt / br + A2D_[1, 0] * bp / br,
            A2D_[0, 1] * bt / br + A2D_[1, 1] * bp / br,
            A2D_[0, 2] * bt / br + A2D_[1, 2] * bp / br,
        ]
    ]
)
A3D_ = sp.Matrix.vstack(Ar, A2D_)

# if PRINT_LATEX:
#    print('E = X (Jth, Jph, 1):')
#    print(sp.latex(A3D_))

# Check that A3D_ multiplied by [x, y, 1] vector is perpendicular to b.
assert (A3D_ * sp.Matrix([[sp.symbols("x")], [sp.symbols("y")], [1]])).dot(
    sp.Matrix([br, bt, bp])
).simplify() == 0


ee = sp.Matrix([[e1r, e1t, e1p], [e2r, e2t, e2p]])
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
    print(
        "\\alpha_{11}=&"
        + "\\eta_P("
        + sp.latex(alpha11.coeff(eP))
        + ")    && + \\eta_H("
        + sp.latex(alpha11.coeff(eH))
        + ")\\nonumber \\\\"
    )
    print(
        "=&                 "
        "\\eta_P\\alpha_{11}^{\\eta_P}                    &&"
        " + \\eta_H\\alpha_{11}^{\\eta_H}                         \\\\"
    )
    print(
        "\\alpha_{12}=&"
        + "\\eta_P("
        + sp.latex(alpha12.coeff(eP))
        + ")    && + \\eta_H("
        + sp.latex(alpha12.coeff(eH))
        + ")\\nonumber \\\\"
    )
    print(
        "=&                 "
        "\\eta_P\\alpha_{12}^{\\eta_P}                    &&"
        " + \\eta_H\\alpha_{12}^{\\eta_H}                         \\\\"
    )
    print(
        "\\alpha_{21}=&"
        + "\\eta_P("
        + sp.latex(alpha21.coeff(eP))
        + ")    && + \\eta_H("
        + sp.latex(alpha21.coeff(eH))
        + ")\\nonumber \\\\"
    )
    print(
        "=&                 "
        "\\eta_P\\alpha_{21}^{\\eta_P}                    &&"
        " + \\eta_H\\alpha_{21}^{\\eta_H}                         \\\\"
    )
    print(
        "\\alpha_{22}=&"
        + "\\eta_P("
        + sp.latex(alpha22.coeff(eP))
        + ")    && + \\eta_H("
        + sp.latex(alpha22.coeff(eH))
        + ")\\nonumber \\\\"
    )
    print(
        "=&                 "
        "\\eta_P\\alpha_{22}^{\\eta_P}                    &&"
        " + \\eta_H\\alpha_{22}^{\\eta_H}                         \\\\"
    )
    print(
        "\\alpha_{13}=&"
        + "u_\\theta("
        + sp.latex(alpha23.coeff(ut))
        + ")    && + u_\\phi("
        + sp.latex(alpha23.coeff(up))
        + ")\\nonumber \\\\"
    )
    print(
        "=&                 "
        "u_\\theta\\alpha_{13}^{u_\\theta}                &&"
        " + u_\\phi\\alpha_{13}^{u_\\phi}                         \\\\"
    )
    print(
        "\\alpha_{23}=&"
        + "u_\\theta("
        + sp.latex(alpha23.coeff(ut))
        + ")    && + u_\\phi("
        + sp.latex(alpha23.coeff(up))
        + ")\\nonumber \\\\"
    )
    print(
        "=&                 "
        "u_\\theta\\alpha_{23}^{u_\\theta}                &&"
        " + u_\\phi\\alpha_{23}^{u_\\phi}                             "
    )


if not PRINT_LATEX:  # print code:
    print("CODE:\n\n")
    print("# resistance terms:")
    print("alpha11_eP = " + str(alpha11.coeff(eP)))
    print("alpha12_eP = " + str(alpha12.coeff(eP)))
    print("alpha21_eP = " + str(alpha21.coeff(eP)))
    print("alpha22_eP = " + str(alpha22.coeff(eP)))
    print("alpha11_eH = " + str(alpha11.coeff(eH)))
    print("alpha12_eH = " + str(alpha12.coeff(eH)))
    print("alpha21_eH = " + str(alpha21.coeff(eH)))
    print("alpha22_eH = " + str(alpha22.coeff(eH)))

    print("\n# wind terms:")
    print("alpha13_ut = " + str(alpha13.coeff(ut)))
    print("alpha23_ut = " + str(alpha23.coeff(ut)))
    print("alpha13_up = " + str(alpha13.coeff(up)))
    print("alpha23_up = " + str(alpha23.coeff(up)))
