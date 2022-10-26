#! /usr/bin/python3
"""
Are you tired of simulations struggling with numerically unstable time steps ?
Have you finally realized that 0.1 is a _really_ bad time step choice, but
don't know how to do better ?

Your ills will finally be soothed !

Give `find_sane_dt` the time step you think you want, and it will reply
with the time step you _actually_ want, i.e. one with an exact representation
on our beloved von Neumann architectures.

Rejoice !

>>> find_sane_dt(.125)
dt ≈ 0.125 = 2⁻³
0.125

>>> find_sane_dt(.1)
dt ≈ 0.09999942779541016 = 2⁻⁴ + 2⁻⁵ + 2⁻⁸ + 2⁻⁹ + 2⁻¹² + 2⁻¹³ + 2⁻¹⁶ + 2⁻¹⁷ + 2⁻²⁰
0.09999942779541016

>>> find_sane_dt(.001)
dt ≈ 0.0009999945759773254 = 2⁻¹⁰ + 2⁻¹⁶ + 2⁻¹⁷ + 2⁻²¹ + 2⁻²⁴ + 2⁻²⁷
0.0009999945759773254

Author: Alexandre René
Source: https://gist.github.com/alcrene/e23be98e33f86e2d3ccfb9dfca5cf1c7
License: MIT
"""

import math
def find_sane_dt(dt, tol=1e-5, show=True):
    "Return a dt as close as possible to `dt` which is exact in binary"
    new_dt = 0
    powers = []
    while (dt-new_dt)/dt > tol:
        powers.append(math.floor(math.log2(dt-new_dt)))
        new_dt = sum(2**p for p in powers)
    if show:
        s = " + ".join(f"2{make_int_superscript(p)}" for p in powers)
        print(f"dt ≈ {new_dt} = {s}")
    return new_dt

def make_int_superscript(v: int) -> str:
    """
    Convert an integer to a string of superscript digits.
    Negative values are supported.
    """
    exponents = list("⁰¹²³⁴⁵⁶⁷⁸⁹")
    s = []
    if v == 0:
        return "⁰"
    elif v < 0:
        s.append("⁻")
        v = -v
    for d in str(v):
        s.append(exponents[ord(d)-ord("0")])
    return "".join(s)

if __name__ == "__main__":
    import sys
    dt = float(sys.argv[1])
    find_sane_dt(dt)
