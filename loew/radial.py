import numpy as np


def vslit_zip(z, dt, du):
    edu = np.exp(1j * du)
    D = np.exp(dt + 1j * du) * (z + 1.0) ** 2 / z
    C = np.sqrt(D ** 2 - 4.0 * edu * D)
    w = 0.5 * (C - 2.0 * edu + D)
    mask = np.abs(w) > 1.0
    w[mask] -= C[mask]
    return w


def trace(times, drive):
    N = times.size
    trace = np.ones(N, dtype=np.complex128)
    for i in range(N - 1, 0, -1):
        dt = times[i] - times[i - 1]
        du = drive[i] - drive[i - 1]
        trace[i:] = vslit_zip(trace[i:], dt, du)
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
