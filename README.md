LOEW - Schramm-Loewner Evolutions for Python
--------------------------------------------

`loew` is a package for simulation of Schramm-Loewner Evolutions (SLE).

[SLE](https://en.wikipedia.org/wiki/Schramm%E2%80%93Loewner_evolution) is a
nifty mathematical tool to analyze families of random fractals related to
[critical phenomena](https://en.wikipedia.org/wiki/Critical_phenomena).
In the last few years some work in the area has moved to numerics, so I hope
this package will be of some help to those involved. But even if yours is a
pure mathematics no-computers-allowed type of work, maybe you can use it to make some
sweet figures for your paper.

I've made the package to be simple of use, read and edit. Suggestions and
pull requests are welcome.

Install
-------
This package depends on `numpy` and `numexpr`. If you're using
[anaconda](https://www.continuum.io/downloads) (and you should), You can easily
install them with the command

```
conda install numpy numpexpr
```

Right now you can can use the package by simply downloading the folder to your
working directory and importing it normally. I'm currently working to getting into 
PyPI so you can pip install it.

Example
--------
Be sure to check the [jupyter
notebook](https://github.com/hfcredidio/loew/blob/master/Examples.ipynb) for
more examples.

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
![](https://raw.githubusercontent.com/hfcredidio/loew/master/images/exmaple.png)

References
----------
Kager, Wouter, and Bernard Nienhuis. "A guide to stochastic Löwner evolution
and its applications."  
Journal of statistical physics 115.5-6 (2004):
1149-1229.

Kennedy, Tom. "Numerical computations for the Schramm-Loewner evolution."  
Journal of Statistical Physics 137.5-6 (2009): 839-856.

License
-------
MIT
