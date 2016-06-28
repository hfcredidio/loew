import numpy as np
import numexpr as ne


def vslit_zip(z, dt, du):
    return ne.evaluate("1j * sqrt(4 * dt - z ** 2) + du")


def vslit_unzip(z, x, y):
    return ne.evaluate("1j * sqrt(-(z - x) ** 2 - y * y)")


def trace(t, u):
    n = len(t)
    z = np.zeros(n, dtype=np.complex128)
    for i in range(n - 1, 0, -1):
        dt = t[i] - t[i - 1]
        du = u[i] - u[i - 1]
        z[i:] = vslit_zip(z[i:], dt, du)
    return z


def drive(z, destroy=False):
    if not destroy:
        z = np.copy(z)
    for i, w in enumerate(z[1:], start=1):
        z[i + 1:] = vslit_unzip(z[i + 1:], w.real, w.imag)
    u = np.cumsum(z.real)
    t = np.cumsum(z.imag ** 2 * 0.25)
    return t, u

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from misc import brownian_motion
    np.random.seed(100)
    t = np.linspace(0.0, 10.0, 1000)
    u = brownian_motion(t, kappa=2.0)
    z = trace(t, u)

    fig, ax = plt.subplots()
    ax.plot(z.real, z.imag)
    plt.show()
