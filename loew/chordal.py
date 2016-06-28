import numpy as np
import numexpr as ne


def vslit_zip(z, dt, du):
    '''
    Vertical slit discretication of chordal Loewner chains. This function is
    the inverse of the solution of Loewner's Equation with driving function

    U(t) = du, for 0 <= t <= dt

    It maps the origin to the point w = du + 2j * dt**0.5

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
    return ne.evaluate("1j * sqrt(4 * dt - z ** 2) + du")


def vslit_unzip(z, x, y):
    '''
    Vertical slit discretication of chordal Loewner chains. This function is
    the solution of Loewner's Equation with driving function

    U(t) = du, for 0 <= t <= dt

    Parameters
    ----------
    z: complex or ndarray of complex
        Input Array.

    x: float
    y: float
        Vertical slit parameters.

    Returns
    -------
    w: complex or ndarray of complex
        Same shape as input array.
    '''
    return ne.evaluate("1j * sqrt(-(z - x) ** 2 - y * y)")


def trace(t, u):
    '''
    Compute the discretized Loewner trace of a driving function u(t) sampled at
    discreet time instants. The trace is defined as a curve z(t) such that
    z(t)= f(t, u(t)) where f(t, w) is the inverse of g(t, w), which is the
    solution of the Loewner Equation in the upper half-plane

    dg          2
    -- = ----------------
    dt    g(t, w) - u(t)

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

    Returns
    -------
    z: 1d ndarrays of complex
        Points of the trace at each time instant.
    '''
    n = len(t)
    z = np.zeros(n, dtype=np.complex128)
    for i in range(n - 1, 0, -1):
        dt = t[i] - t[i - 1]
        du = u[i] - u[i - 1]
        z[i:] = vslit_zip(z[i:], dt, du)
    return z


def drive(z, destroy=False):
    '''
    Compute the driving function u(t) of a Loewner trace z.
    The drive is obtained by inverting the zipper algorithm.
    Because of the choice of capacity a(t) = 2t, the time
    is obtained directly from the trace.

    Reference
    ---------
    Kennedy, Tom. "Numerical computations for the Schramm-Loewner evolution."
    Journal of Statistical Physics 137.5-6 (2009): 839-856.

    Parameters
    ----------
    z: 1d ndarrays of complex
        Points of the discretized trace.

    destroy: bool, optional
        If set to False, the routine make a copy of z before performing the
        algorithm. Otherwise it destroys the information on the trace, but
        saves memory. Default if False.

    Returns
    -------
    t: 1d ndarray of floats
        Time instants where of each trace point. Same shape as z.

    u: 1d ndarray of floats
        Values of the driving function at each point of the trace.
        Same shape as z.
    '''
    if not destroy:
        z = np.copy(z)
    for i, w in enumerate(z[1:], start=1):
        z[i + 1:] = vslit_unzip(z[i + 1:], w.real, w.imag)
    u = np.cumsum(z.real)
    t = np.cumsum(z.imag ** 2 * 0.25)
    return t, u
