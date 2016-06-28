LOEW - Schramm-Loewner Evolutions for Python
--------------------------------------------

`loew` is a python packages for simulations of
Schramm-Loewner evolutions (SLE).

SLE is a nifty mathematical tool used .


Examples
--------

```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import loew

np.random.seed(123)
t = np.linspace(0.0, 1.0, 10000)
u = loew.misc.brownian_motion(t, kappa=2)
z = loew.chordal.trace(t, u)

plt.plot(z.real, z.imag, lw=2)
plt.gca().set_aspect('equal')
plt.show()
```
![](http://files.te52.com/testtalk/files/2015/04/testing.png)

References
----------
Kager, Wouter, and Bernard Nienhuis. "A guide to stochastic Löwner evolution
and its applications." Journal of statistical physics 115.5-6 (2004):
1149-1229.

Kennedy, Tom. "Numerical computations for the Schramm-Loewner evolution."
Journal of Statistical Physics 137.5-6 (2009): 839-856.

License
-------
MIT
