import numpy as np

def brownian_motion(t, kappa=1):
    dt = np.ediff1d(t, to_begin=t[0])
    du = np.random.randn(len(t)) * np.sqrt(dt * kappa)
    u = np.add.accumulate(du)
    return u

def fractional_brownian_motion(n, h, b=1.0, tf=1.0):
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
