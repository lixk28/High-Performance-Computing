DFT:
$$
\displaystyle Y[p] = \sum_{k=0}^{n-1}X[k]w^{kp}, \quad \displaystyle w = e^\frac{2\pi i}{n}
$$
FFT:

$w = e^\frac{2\pi i}{n} = cos\frac{2\pi}{n} + isin\frac{2\pi}{n}$
$$
\begin{align}
Y[p] &= \sum_{k=0}^{n/2-1}X[2k]w^{2kp} + \sum_{k=0}^{n/2-1}X[2k+1]w^{(2k+1)p} \quad (even + odd) \\
&= \sum_{k=0}^{n/2-1}X[2k]w^{2kp} + \sum_{k=0}^{n/2-1}X[2k+1]w^pw^{2kp} \\
&= \sum_{k=0}^{n/2-1}X[2k]\hat{w}^{kp} + w^p\sum_{k=0}^{n/2-1}X[2k+1]\hat{w}^{kp} \quad (\hat{w} = e^{2\pi i/(n/2) \times kp} = w^2) \\
\end{align}
$$
