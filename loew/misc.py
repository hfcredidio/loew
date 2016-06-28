import numpy as np


def brownian_motion(t, kappa=1.0):
    '''
    Compute the 1D Brownian motion at some given time instants. The Brownian
    motion is such that each increment B(t[i]) - B(t[i-1]) are independent
    random variables normally distributed with zero mean an kappa*(t[i]-t[i-1])
    variance.

    Parameters
    ----------
    t : 1d ndarray of floats
        Time instants where the Brownian motion is to be evaluated.

    kappa : float, optional
        Diffusion constant. Default is 1.0.

    Returns
    -------
    u : 1d ndarray of floats
        1D Brownian motion evaluated. Same shape as t.
    '''
    dt = np.ediff1d(t, to_begin=t[0])
    du = np.random.randn(len(t)) * np.sqrt(dt * kappa)
    u = np.add.accumulate(du)
    return u


def fractional_brownian_motion(n, h, b=1.0, tf=1.0):
    '''
    Compute the 1D fractional Brownian motion. These are stochastic processes
    with long range power-law correlations. The mean square displacement
    behaves as

    <B(t)^2> = b t^(2h)

    where b is the diffusion constant and h is the Hurst exponent.

    It uses the Davies-Harte algorithm.

    Reference
    ---------
    Davies, Robert B., and D. S. Harte. "Tests for Hurst effect."
    Biometrika 74.1 (1987): 95-101.

    Parameters
    ----------
    n : int
        Number of point the function will produce.

    h : float
        Hurst exponent.

    b : float, optional
        Diffusion constant. Default is 1.0.

    tf : float, optional
        Endind time, so the brownian motion is evaluated at
        n time instants uniformly spaced in the interval [0, tf].
        Default if 1.0.

    Returns
    -------
    t : ndarray of floats
        Time instants where the fractional Brownian motion was evaluated.

    X : ndarray of floats
        Values of fractional Brownian motion.
    '''
    n2 = 2 * n
    h2 = 2 * h

    i = np.arange(0, n+1)
    s = np.zeros(n2)
    s[:n+1] = np.abs(i+1) ** h2 + np.abs(i-1) ** h2 - 2 * np.abs(i) ** h2
    s[:n+1] *= b * 0.5
    s[n+1:] = s[n-1:0:-1]

    A = np.fft.fft(s).real
    assert (A >= 0).all()

    Y = np.empty(n2, dtype=np.complex128)
    Y[0] = 2**0.5 * np.random.randn()
    Y[n] = 2**0.5 * np.random.randn()
    Y[1:n] = np.random.randn(n-1) + 1j * np.random.randn(n-1)
    Y[n+1:n2] = Y[n-1:0:-1].conj()
    Y *= np.sqrt(A*n)

    X = np.fft.ifft(Y).real[:n]
    X = np.cumsum(X)
    X -= X[0]

    t = np.linspace(0.0, tf, n)
    X *= (tf / n) ** h

    return t, X
