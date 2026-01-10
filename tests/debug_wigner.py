
import numpy as np
from pynamit.spherical_harmonics.wigner import wigner_3j

def debug_calc():
    l1, l2, l3 = 1, 1, 0
    m1, m2, m3 = 0, 0, 0
    s1, s2, s3 = 1, 1, 0
    
    # Inputs to Gaunt (Bra conjugated)
    m1_g, s1_g = -m1, -s1
    
    print(f"Inputs: l1={l1} s1={s1} m1={m1} | Conjugated: ({m1_g}, {s1_g})")
    
    # Wigner 3j M
    w3j_m = wigner_3j(l1, l2, l3, m1_g, m2, m3)
    print(f"W3j M (0,0,0): {w3j_m}")
    
    # Wigner 3j S
    w3j_s = wigner_3j(l1, l2, l3, s1_g, s2, s3)
    print(f"W3j S (-1,1,0): {w3j_s}")
    
    # Pref
    pref = np.sqrt( (2*l1+1)*(2*l2+1)*(2*l3+1) / (4*np.pi) )
    print(f"Pref: {pref}")
    
    g = pref * w3j_m * w3j_s
    print(f"Gaunt g: {g}")
    
    phase = (-1.0)**(m1 + s1)
    print(f"Phase (-1)^(m+s): {phase}")
    
    total = g * phase
    print(f"Total Factor: {total}")

if __name__ == "__main__":
    debug_calc()
