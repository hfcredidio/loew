import numpy as np
import numexpr as ne


def vslit_zip(z, dt, du, width=1.0):
    '''
    Vertical slit discretication of dipolar Loewner chains. This function is
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
    w2 = 2 * width
    w2  # just so pep8 stop bitching
    return ne.evaluate("w2 * 1j * arccos(cosh(z / w2) * "
                       "exp(-dt / (w2 * width))) + du")


def trace(t, u, width=1.0):
    '''
    Compute the discretized Loewner trace of a driving function u(t) sampled at
    discreet time instants. The trace is defined as a curve z(t) such that
    z(t)= f(t, u(t)) where f(t, w) is the inverse of g(t, w), which is the
    solution of the Loewner Equation in the infinite strip of width pi*D

    dg                1 / D
    -- = -------------------------------
    dt    tanh{ [g(t, z) - u(t)] / 2D }

    It makes use of the zipper algorithm with a vertical slit discretization
    scheme.

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

    width: float, optional
        The width of the strip is width*pi. Default is 1.0.

    Returns
    -------
    z: 1d ndarrays of complex
        Points of the trace at each time instant.
    '''
    n = t.size
    trace = np.zeros(n, dtype=np.complex128)
    for i in range(n - 1, 0, -1):
        dt = t[i] - t[i - 1]
        du = u[i] - u[i - 1]
        trace[i:] = vslit_zip(trace[i:], dt, du, width)
    return trace
