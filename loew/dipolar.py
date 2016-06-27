import numpy as np
import numexpr as ne


def zero_map(z, dt, du, width=1.0):
    w2 = 2 * width
    w2  # just so pep8 stop bitching
    return ne.evaluate("w2 * 1j * arccos(cosh(z / w2) * "
                       "exp(-dt / (w2 * width))) + du")


def trace(times, drive, width=1.0):
    N = times.size
    trace = np.zeros(N, dtype=np.complex128)
    for i in range(N - 1, 0, -1):
        dt = times[i] - times[i - 1]
        du = drive[i] - drive[i - 1]
        trace[i:] = zero_map(trace[i:], dt, du, width)
    return trace


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
