import numpy as np


def vslit_zip(z, dt, du):
    '''
    Vertical slit discretication of radial Loewner chains. This function is
    the inverse of the solution of Loewner's Equation with driving function

    U(t) = du, for 0 <= t <= dt

    Parameters
    ----------
    z : complex or ndarray of complexes
        Input array.

    dt: float
    du: float
        Vertical slit parameters.

    Returns
    -------
    w : complex or ndarray of complxes
        Same shape as input array.

    '''
    edu = np.exp(1j * du)
    D = np.exp(dt + 1j * du) * (z + 1.0) ** 2 / z
    C = np.sqrt(D ** 2 - 4.0 * edu * D)
    w = 0.5 * (C - 2.0 * edu + D)
    mask = np.abs(w) > 1.0
    w[mask] -= C[mask]
    return w


def trace(t, u):
    '''
    Compute the discretized Loewner trace of a driving function u(t) sampled at
    discreet time instants. The trace is defined as a curve z(t) such that
    z(t)= f(t, u(t)) where f(t, w) is the inverse of g(t, w), which is the
    solution of the Loewner Equation in the unit disk

    dg            exp{i u(t)} + g(t, w)
    -- = g(t, w) -----------------------
    dt            exp{i u(t)} - g(t, w)

    It makes use of the zipper algorithm with a vertical slit discretization
    scheme.

    It is considerably slower than the chordal and dipolar versions.

    Reference
    ---------
    Kennedy, Tom. "Numerical computations for the Schramm-Loewner evolution."
    Journal of Statistical Physics 137.5-6 (2009): 839-856.

    Parameters
    ----------
    t: 1d ndarray of floats
        Time instants where the driving is sampled.

    u: 1d ndarray of floats
        Sampled values of the driving function.

    Returns
    -------
    z: 1d ndarrays of complex
        Points of the trace at each time instant.
    '''
    n = t.size
    z = np.ones(n, dtype=np.complex128)
    for i in range(n - 1, 0, -1):
        dt = t[i] - t[i - 1]
        du = u[i] - u[i - 1]
        z[i:] = vslit_zip(z[i:], dt, du)
    return z
