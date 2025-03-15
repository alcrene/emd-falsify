---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: py:percent,md:myst
  notebook_metadata_filter: -jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python (emd-falsify-dev)
  language: python
  name: emd-falsify-dev
---

+++ {"tags": ["remove-cell"], "editable": true, "slideshow": {"slide_type": ""}}

---
math:
  '\lnLtt': '{q^*}'
  '\EE'   : '{\mathbb{E}}'
  '\VV'   : '{\mathbb{V}}'
  '\nN'   : '{\mathcal{N}}'
  '\Mvar' : '{\mathop{\mathrm{Mvar}}}'
  '\Beta' : '{\mathop{\mathrm{Beta}}}'
  '\pathP': '{\mathop{\mathfrak{Q}}}'
  '\lnLh' : '{\hat{q}}'
  '\emdstd': '{\sqrt{c} δ^{\mathrm{EMD}}}'
  '\EMD'  : '{\mathrm{EMD}}'
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(supp_path-sampling)=
# Sampling quantile paths

+++ {"editable": true, "slideshow": {"slide_type": ""}}

```{only} html
{{ prolog }}
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
# import jax
# import jax.numpy as jnp
# from functools import partial
# jax.config.update("jax_enable_x64", True)
```

```{code-cell}
:tags: [hide-input]

import logging
import warnings
import math
import time
#import multiprocessing as mp
import numpy as np
import scipy.special
import scipy.optimize
#from scipy.special import digamma, polygamma
#from scipy.optimize import root, brentq
from tqdm.auto import tqdm

from collections.abc import Callable
from pathlib import Path
from typing import Optional, Union, Literal, Tuple, Generator
from scityping import Real
from scityping.numpy import Array, Generator as RNGenerator

from emd_falsify.digitize import digitize  # Used to improve numerical stability when finding Beta parameters
```

+++ {"tags": ["remove-cell"]}

Notebook-only imports

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
import itertools
import scipy.stats
import holoviews as hv
from myst_nb import glue

from emd_falsify.utils import GitSHA
from config import config  # Uses config from CWD

hv.extension(config.viz.backend)
```

```{code-cell}
:tags: [remove-cell]

logger = logging.getLogger(__name__)
```

```{code-cell}
__all__ = ["generate_quantile_paths", "generate_path_hierarchical_beta"]
```

We want to generate paths $\lnLh$ for the quantile function $l(Φ)$, with $Φ \in [0, 1]$, from a stochastic process $\pathP$ determined by $\lnLtt(Φ)$ and $\emdstd(Φ)$. This process must satisfy the following requirements:
- It must generate only monotone paths, since quantile functions are monotone.
- The process must be heteroscedastic, with variability at $Φ$ given by $\emdstd(Φ)$.
- Generated paths should track the function $\lnLtt(Φ)$, and track it more tightly when variability is low.
- Paths should be “temporally uncorrelated”: each stop $Φ$ corresponds to a different ensemble of data points, which can be drawn in any order. So we don't expect any special correlation between $\lnLh(Φ)$ and $\lnLh(Φ')$, beyond the requirement of monotonicity.
  + In particular, we want to avoid defining a stochastic process which starts at one point and accumulates variance, like the $\sqrt{t}$ envelope characteristic of a Gaussian white noise.
  + Concretely, we require the process to be “$Φ$-symmetric”: replacing $\lnLtt(Φ) \to \lnLtt(-Φ)$ and $\emdstd(Φ) \to \emdstd(-Φ)$ should define the same process, just inverted along the $Φ$ axis.

+++

(code_path-sampling_hierarchical-beta)=
## Hierarchical beta process

+++

Because the joint requirements of monotonicity, non-stationarity and $Φ$-symmetry are uncommon for a stochastic process, some care is required to define an appropriate $\pathP$. The approach we choose here is to first select the end points $\lnLh(0)$ and $\lnLh(1)$, then fill the interval by successive binary partitions: first $\bigl\{\lnLh\bigl(\tfrac{1}{2}\bigl)\bigr\}$, then $\bigl\{\lnLh\bigl(\tfrac{1}{4}\bigr), \lnLh\bigl(\tfrac{3}{4}\bigr)\bigr\}$, $\bigl\{\lnLh\bigl(\tfrac{1}{8}\bigr), \lnLh\bigl(\tfrac{3}{8}\bigr), \lnLh\bigl(\tfrac{5}{8}\bigr), \lnLh\bigl(\tfrac{7}{8}\bigr)\bigr\}$, etc. (Below we will denote these ensembles $\{\lnLh\}^{(1)}$, $\{\lnLh\}^{(2)}$, $\{\lnLh\}^{(3)}$, etc.) Thus integrating over paths becomes akin to a path integral with variable end points.
Moreover, instead of drawing quantile values, we draw increments
```{math}
:label: eq_def-quantile-increment
Δ q_{ΔΦ}(Φ) := \lnLh(Φ+ΔΦ) - \lnLh(Φ) \,.
```
Given two initial end points $\lnLh(0)$ and $\lnLh(1)$, we therefore first we draw the pair $\bigl\{Δ q_{2^{-1}}(0),\; Δ q_{2^{-1}}\bigl(2^{-1}\bigr)\}$, which gives us
```{math}
\lnLh\bigl(2^{-1}\bigr) = \lnLh(0) + Δ q_{2^{-1}}(0) = \lnLh(1) - Δ q_{2^{-1}}\bigl(2^{-1}\bigr)\,.
```
Then $\bigl\{\lnLh(0), \lnLh\bigl(\frac{1}{2}\bigr) \bigr\}$ and $\bigl\{ \lnLh\bigl(\frac{1}{2}\bigr), \lnLh(1) \bigr\}$ serve as end points to draw $\bigl\{Δ q_{2^{-2}}\bigl(0\bigr),\; Δ q_{2^{-2}}\bigl(2^{-2}\bigr) \bigr\}$ and $\bigl\{Δ q_{2^{-2}}\bigl(2^{-1}\bigr),\; Δ q_{2^{-2}}\bigl(2^{-1} + 2^{-2}\bigr) \bigr\}$. We repeat the procedure as needed, sampling smaller and smaller incremenents, until the path has the desired resolution. As the increments are constrained:
```{math}
Δ q_{2^{-n}}(Φ) \in \bigl( 0, \lnLh(Φ+2^{-n+1}) - \lnLh(Φ)\,\bigr)\,,
```
the path thus sampled is always monotone. Note also that increments must be drawn in pairs (or more generally as a *combination*) of values constrained by their sum:
```{math}
:label: eq_sum-constraint
Δ q_{2^{-n}}\bigl(Φ\bigr) + Δ q_{2^{-n}}\bigl(Φ + 2^{-n} \bigr) \stackrel{!}{=} \lnLh(Φ+2^{-n+1}) - \lnLh(Φ) \,.
```
The possible increments therefore lie on a 1-simplex, for which a natural choice is to use a beta distribution[^1], with the random variable corresponding to the first increment $Δ q_{2^{-n}}(Φ)$. The density function of a beta random variable has the form
```{math}
:label: eq_beta-pdf
p(x_1) \propto x^{α-1} (1-x)^{β-1}\,,
```
with $α$ and $β$ parameters to be determined.

+++

:::{important}  
An essential property of a stochastic process is *consistency*: it must not matter exactly how we discretize the interval {cite:p}`gillespieMathematicsBrownianMotion1996`. Let $\{\lnLh\}^{(n)}$ denote the steps which are added when we refine the discretization from steps of $2^{-n+1}$ to steps of $2^{-n}$:
```{math}
:label: eq_added-steps
\{\lnLh\}^{(n)} := \bigl\{\lnLh(k\cdot 2^{-n}) \,\big|\, k=1,3,\dotsc,2^n \bigr\} \,.
```
A necessary condition for consistency is that coarsening the discretization from steps of $2^{-n}$ to steps of $2^{-n+1}$ (i.e. marginalizing over the points at $\{\lnLh\}^{(n)}$) does not substantially change the probability law:
```{math}
:label: eq_consistency-condition
p\bigl(\{\lnLh\}^{(n)}\bigr)\bigr) \stackrel{!}{=} \int p\bigl(\{\lnLh\}^{(n)} \,\big|\, \{\lnLh\}^{(n+1)}\bigr) \,d\{\lnLh\}^{(n+1)} \;+\; ε\,,
```
with $ε$ vanishing as $n$ is increased to infinity.

Satisfying this requirement is required in order to compute integrals over $\lnLh$, since otherwise they don’t converge as $ΔΦ$ is reduced.
:::

+++

[^1]: One could conceivably draw all increments at once, with a [*shifted scaled Dirichlet distribution*](https://doi.org/10.1007/978-3-030-71175-7_4) instead of a beta, if it can be shown that also in this case coarsening the distribution still results in the same probability law.

+++

(supp_path-sampling_conditions-beta-param)=
### Conditions for choosing the beta parameters

:::{margin}
Recall that we made the assumption that the variability of the path process $\pathP$ should determined by $δ^{\EMD}$, up to some constant $c$.{cite:p}`reneFalsifyingModels2024` This constant is determined by a calibration experiment.
To keep expressions concise, in this section we use $\emdstd(Φ) := c δ^{\EMD}(Φ)$.
:::
To draw an increment $Δ q_{2^{-n}}$, we need to convert $\lnLtt(Φ)$ and $\emdstd(Φ)$ into beta distribution parameters $α$ and $β$. If $x_1$ follows a beta distribution, then its first two cumulants are given by
```{math}
\begin{aligned}
x_1 &\sim \Beta(α, β) \,, \\
\EE[x_1] &= \frac{α}{α+β} \,, \\
\VV[x_1] &= \frac{αβ}{(α+β)^2(α+β+1)} \,. \\
\end{aligned}
```
However, as discussed by Mateu-Figueras et al. (2021, 2011), to properly account for the geometry of a simplex, one should use instead statistics with respect to the Aitchison measure, sometimes referred to as the *center* and *metric variance*. Defining $x_2 = 1-x_1$, these can be written (Mateu-Figueras et al., 2021)
```{math}
:label: eq_Aitchison-moments

\begin{align}
\EE_a[(x_1, x_2)] &= \frac{1}{e^{ψ(α)} + e^{ψ(β)}} \bigl[e^{ψ(α)}, e^{ψ(β)}\bigr] \,, \\
\Mvar[(x_1, x_2)] &= \frac{1}{2} \bigl(ψ_1(α) + ψ_1(β)\bigr) \,. \\
\end{align}
```
Here $ψ$ and $ψ_1$ are the digamma and trigamma functions respectively.
(In addition to being more appropriate, the center and metric variance are also better suited for defining a consistent stochastic process. For example, since the metric variance is unbounded, we can always scale it with $\emdstd(Φ)$ without exceeding its domain.)

+++

Since we want the sum to be $d := \lnLh(Φ+2^{-n+1}) - \lnLh(Φ)$, we define
```{math}
:label: eq_relation-beta-increment
\bigl[Δ q_{2^{-n}}\bigl(Φ\bigr),\, Δ q_{2^{-n}}\bigl(Φ+2^{-n})\bigr)\bigr] = d \bigl[x_1, x_2\bigr] \,.
```
Then

+++

$$\begin{aligned}
\EE_a\Bigl[\bigl[Δ q_{2^{-n}}\bigl(Φ\bigr),\, Δ q_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]\Bigr] &= \frac{d}{e^{ψ(α)} + e^{ψ(β)}} \bigl[e^{ψ(α)}, e^{ψ(β)}\bigr] \,, \\
\Mvar\Bigl[\bigl[Δ q_{2^{-n}}\bigl(Φ\bigr),\, Δ q_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]\Bigr] &= \frac{1}{2} \bigl(ψ_1(α) + ψ_1(β)\bigr) \,.
\end{aligned}$$

+++

We now choose to define the parameters $α$ and $β$ via the following relations:
:::{admonition} &nbsp;
:class: important

$$\begin{aligned}
\EE_a\Bigl[\bigl[Δ q_{2^{-n}}\bigl(Φ\bigr),\, Δ q_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]\Bigr] &\stackrel{!}{=}^* \bigl[\, \lnLtt\bigl(Φ+2^{-n}\bigr) - \lnLtt\bigl(Φ\bigr),\,\lnLtt\bigl(Φ+2^{-n+1}\bigr) - \lnLtt\bigl(Φ+2^{-n}\bigr) \,\bigr]\,, \\
\Mvar\Bigl[\bigl[Δ q_{2^{-n}}\bigl(Φ\bigr),\, Δ q_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]\Bigr] &\stackrel{!}{=} \emdstd\bigl(Φ+2^{-n}\bigr)^2 \,.
\end{aligned}$$ (eq_defining-conditions-a)
:::

+++

These follow from interpretating $\lnLtt$ and $\emdstd$ as estimators for the center and square root of the metric variance.
We use $=^*$ to indicate equality in spirit rather than true equality, since strictly speaking, these are 3 equations for 2 unknown. To reduce the $\EE_a$ equations to one, we use instead

:::{admonition} &nbsp;
:class: important

$$\frac{\EE_a\bigl[Δ q_{2^{-n}}\bigl(Φ\bigr)\bigr]}{\EE_a \bigl[Δ q_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]} \stackrel{!}{=} \frac{\lnLtt\bigl(Φ+2^{-n}\bigr) - \lnLtt\bigl(Φ\bigr)}{\lnLtt\bigl(Φ+2^{-n+1}\bigr) - \lnLtt\bigl(Φ+2^{-n}\bigr)} \,.$$ (eq_defining_conditions-b)
:::

+++

**Remarks**
- We satisfy the necessary condition for consistency by construction:
  ```{math}
  p\bigl(\{l\}^{(n)}\bigr)\bigr) = \int p\bigl(\{l\}^{(n)} \,\big|\, \{l\}^{(n+1)}\bigr) \,d\{l\}^{(n+1)}\,.
  ```
- The stochastic process is not Markovian, so successive increments are not independent. The variance of a larger increment therefore need not equal the sum of the variance of constituent smaller ones; in other words,
  ```{math}
  Δ q_{2^{-n+1}}\bigl(Φ\bigr) = Δ q_{2^{-n}}\bigl(Φ\bigr) + Δ q_{2^{-n}}\bigl(Φ+2^{-n}\bigr)
  ```
  does *not* imply
  ```{math}
  \VV\bigl[Δ q_{2^{-n+1}}\bigl(Φ\bigr)\bigr] = \VV\bigl[Δ q_{2^{-n}}\bigl(Φ\bigr)\bigr] + \VV\bigl[Δ q_{2^{-n}}\bigl(Φ+2^{-n}\bigr)\bigr]\,.
  ```
- Our defining equations make equivalent use of the pre ($Δ q_{2^{-n}}(Φ)$) and post ($Δ q_{2^{-n}}(Φ+2^{-n})$) increments, thus preserving symmetry in $Φ$.
- Step sizes of the form $2^{-n}$ have exact representations in binary. Thus even small step sizes should not introduce additional numerical errors.

+++

(supp_path-sampling_beta-param-algorithm)=
### Formulation of the parameter equations as a root-finding problem

+++

Define
:::{admonition} &nbsp;
:class: important

$$\begin{align}
r &:= \frac{\lnLtt(Φ+2^{-n}) - \lnLtt(Φ)}{\lnLtt(Φ+2^{-n+1}) - \lnLtt(Φ+2^{-n})} \,; \\
v &:= 2 \emdstd\bigl(Φ + 2^{-n}\bigr)^2 \,.
\end{align}$$ (eq_def-r-v) 
:::
The first value, $r$, is the ratio of two subincrements within $Δ q_{2^{-n+1}}(Φ)$.
Setting $\frac{e^{ψ(α)}}{e^{ψ(β)}} = r$, the two equations we need to solve for $α$ and $β$ can be written
```{math}
:label: eq_root-finding-problem

\begin{align}
ψ(α) - ψ(β) &= \ln r \,; \\
\ln\bigl[ ψ_1(α) + ψ_1(β) \bigr] &= \ln v \,.
\end{align}
```
Note that these equations are symmetric in $Φ$: replacing $Φ$ by $-Φ$ simply changes the sign on both sides of the first. The use of the logarithm in the equation for $v$ helps to stabilize the numerics.

```{code-cell}
def f(lnαβ, lnr_v, _array=np.array, _exp=np.exp, _log=np.log,
      digamma=scipy.special.digamma, polygamma=scipy.special.polygamma):
    α, β = _exp(lnαβ).reshape(2, -1)  # scipy's `root` always flattens inputs
    lnr, v = lnr_v
    return _array((
        digamma(α) - digamma(β) - lnr,
        _log(polygamma(1, α) + polygamma(1, β)) - _log(v)
    )).flat

def f_mid(lnα, v, _exp=np.exp, _log=np.log, polygamma=scipy.special.polygamma):
    "1-d equation, for the special case α=β (equiv to r=1)"
    return _log(2*polygamma(1, _exp(lnα))) - _log(v)
```

```{code-cell}
:tags: [active-ipynb, remove-cell]

# #@partial(jax.jit, static_argnames=("_array", "_exp", "_log", "digamma", "polygamma"))
# def f(lnαβ, lnr_v, _array=jnp.array, _exp=jnp.exp, _log=jnp.log,
#       digamma=jax.scipy.special.digamma, polygamma=jax.scipy.special.polygamma):
#     α, β = _exp(lnαβ).reshape(2, -1)  # scipy's `root` always flattens inputs
#     lnr, v = lnr_v
#     return _array((
#         digamma(α) - digamma(β) - lnr,
#         _log(polygamma(1, α) + polygamma(1, β)) - _log(v)
#     )).flatten()

# #@partial(jax.jit, static_argnames=("_exp", "_log", "polygamma"))
# def f_mid(lnα, v, _exp=jnp.exp, _log=jnp.log, polygamma=jax.scipy.special.polygamma):
#     "1-d equation, for the special case α=β (equiv to r=1)"
#     return _log(2*polygamma(1, _exp(lnα))) - _log(v)
```

:::{margin}  
This implementation of the Jacobian is tested for both scalar and vector inputs, but fits turned out to be both faster and more numerically stable when they don't use it.
Therefore we keep it only for reference and illustration purposes.  
:::

```{code-cell}
:tags: [active-ipynb]

def jac(lnαβ, lnr_v):
    lnα, lnβ = lnαβ.reshape(2, -1)      # scipy's `root` always flattens inputs
    α, β = np.exp(lnαβ).reshape(2, -1)
    j = np.block([[np.diagflat(digamma(α)*lnα), np.diagflat(-digamma(β)*lnβ)],
                     [np.diagflat(polygamma(1,α)*lnα), np.diagflat(polygamma(1,β)*lnβ)]])
    return j
```

The functions $ψ$ and $ψ_1$ diverge at zero, so $α$ and $β$ should remain positive. Therefore it makes sense to fit their logarithm: this enforces the lower bound, and improves the resolution where the derivative is highest. The two objective functions (up to scalar shift) are plotted below: the region for low $\ln α$ and $\ln β$ shows sharp variation around $α=β$, suggesting that this area may be most challenging for a numerical optimizer. In practice this is indeed what we observed.

We found however that we can make fits much more reliable by first choosing a suitable initialization point along the $\ln α = \ln β$ diagonal. In practice this means setting $α_0 = α = β$ and solving the first equation of Eqs. {eq}`eq_root-finding-problem` for $α_0$. (We use the implementation of Brent’s method in SciPy.) Then we can solve the full 2d problem of Eqs. {eq}`eq_root-finding-problem`, with $(α_0, α_0)$ as initial value. This procedure was successful for all values of $r$ and $v$ we encountered in our experiments.

```{code-cell}
:tags: [active-ipynb, hide-input]

α = np.logspace(-2, 1.2)
β = np.logspace(-2, 1.2).reshape(-1, 1)
digamma, polygamma = scipy.special.digamma, scipy.special.polygamma

EEa = digamma(α) / (digamma(α) + digamma(β))
Mvar = 0.5*(polygamma(1, α) + polygamma(1, β))
domλ = np.array([[(ReJ:=np.real(np.linalg.eigvals(jac(np.stack((lnα, lnβ)), 0))))[abs(ReJ).argmax()]
                    for lnβ in np.log(β.flat)]
                   for lnα in np.log(α.flat)])
```

```{code-cell}
:tags: [active-ipynb, hide-input]

dim_lnα = hv.Dimension("lnα", label=r"$\ln α$")
dim_lnβ = hv.Dimension("lnβ", label=r"$\ln β$")
dim_ψα  = hv.Dimension("ψα",  label=r"$ψ(α)$")
dim_ψ1α = hv.Dimension("ψ1α", label=r"$ψ_1(α)$")
dim_Eobj = hv.Dimension("Eobj", label=r"$ψ(α)-ψ(β)$")
dim_lnMvar = hv.Dimension("Mvar", label=r"$\ln \mathrm{{Mvar}}[x_1, x_2]$")  # Doubled {{ because Holoviews applies .format to title
dim_Reλ = hv.Dimension("Reλ", label=r"$\mathrm{{Re}}(λ)$")
fig = hv.Curve(zip(np.log(α.flat), digamma(α.flat)), kdims=[dim_lnα], vdims=[dim_ψα], label=dim_ψα.name).opts(title=dim_ψα.label) \
      + hv.Curve(zip(np.log(α.flat), polygamma(1, α.flat)), kdims=[dim_lnα], vdims=[dim_ψ1α], label=dim_ψα.name).opts(title=dim_ψ1α.label) \
      + hv.QuadMesh((np.log(α.flat), np.log(β.flat), digamma(α)-digamma(β)),
                    kdims=[dim_lnα, dim_lnβ],
                    vdims=[dim_Eobj], label=dim_Eobj.name).opts(title=dim_Eobj.label) \
      + hv.QuadMesh((np.log(α.flat), np.log(β.flat), np.log(Mvar)),
                    kdims=[dim_lnα, dim_lnβ],
                    vdims=[dim_lnMvar], label=dim_lnMvar.name).opts(title=dim_lnMvar.label)
fig.opts(hv.opts.QuadMesh(colorbar=True, clabel=""))
fig.opts(fig_inches=3, sublabel_format="", vspace=0.4, backend="matplotlib")
fig.cols(2);

#glue("fig_polygamma", fig, display=None)
```

```{code-cell}
:tags: [remove-cell, active-ipynb]

#path = config.paths.figures/f"path-sampling_polygamma"
path = Path("path-sampling_polygamma")
hv.save(fig, path.with_suffix(".svg"), backend="matplotlib")
# hv.save(fig, path.with_suffix(".pdf"), backend="matplotlib")
```

:::{figure} path-sampling_polygamma.svg
:name: fig_polygamma

Characterization of the digamma ($ψ$) and trigamma ($ψ_1$) functions, and of the metric variance $\Mvar$.  
:::

+++

Plotting the eigenvalues of the Jacobian (specifically, the real part of its dominant eigenvalue) in fact highlights three regions with a center at roughly $(\ln α, \ln β) = (0, 0)$. (The Jacobian does not depend on $r$ or $v$, so this is true for all fit conditions). 

Note that the color scale is clipped, to better resolve values near zero. Eigenvalues quickly increase by multiple orders of magnitude away from $(0,0)$.

:::{note}
Empirically we found that initializing fits at $(0, 0)$ resulted in robust fits for a large number of $(r,v)$ tuples, even when $r > 100$. We hypothesize that this is because it is difficult for the fit to move from one region to another; by initializing where the Jacobian is small, fits are able to find the desired values before getting stuck in the wrong region.

That being said, there are cases where $(0, 0)$ is not a good initial vector for the root solver is when $\boldsymbol{r \approx 1}$. This can be resolved by choosing a better initial value along the $(α_0, α_0)$ diagonal, as described above. In practice we found no detriment to always using the 1d problem to select an initial vector, so we use that approach in all cases.
:::

```{code-cell}
:tags: [active-ipynb, hide-input]

fig = hv.QuadMesh((np.log(α.flat), np.log(β.flat), domλ),
            kdims=[dim_lnα, dim_lnβ], vdims=[dim_Reλ],
            label="Real part of dom. eig val"
           ).opts(clim=(-1, 1), cmap="gwv",
                  colorbar=True)
#glue("fig_Jac-spectrum", fig, display=False)
```

```{code-cell}
:tags: [remove-cell, active-ipynb]

#path = config.paths.figures/f"path-sampling_jac-spectrum"
path = Path("path-sampling_jac-spectrum")
hv.save(fig, path.with_suffix(".svg"), backend="matplotlib")
#hv.save(fig, path.with_suffix(".pdf"), backend="matplotlib")
```

:::{figure} path-sampling_jac-spectrum.svg
:name: fig_Jac-spectrum

**Objective function has a saddle-point around (0,0)**
After rewriting Eqs. {eq}`eq_root-finding-problem` in terms of $\ln α$ and $\ln β$, we compute the Jacobian $J$. Plotted is the real part of the eigenvalue $λ_i$ of $J$ for which $\lvert\mathop{\mathrm{Re}}(λ)\rvert$ is largest; this gives an indication of how quickly the fit moves away from a given point.
In most cases, a root finding algorithm initialized at (0,0) will find a solution.
:::

+++

(supp_path-sampling_beta-param-special-cases)=
### Special cases for extreme values

For extreme values of $r$ or $v$, the beta distribution becomes degenerate and numerical optimization may break. We identified four cases requiring special treatment.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card}
:columns: 12 8 8 8

$\boldsymbol{r \to 0}$
^^^

The corresponds to stating that $Δ q_{2^{-n}}(Φ)$ is infinitely smaller than $Δ q_{2^{-n}}(Φ+2^{-n})$. Thus we set $x_1 = 1$, which is equivalent to setting

$$\begin{aligned}
Δ q_{2^{-n}}(Φ) &= 0 \,, \\
Δ q_{2^{-n}}(Φ+2^{-n}) &= \lnLtt(Φ+2^{-n+1}) - \lnLtt(Φ) \,.
\end{aligned}$$

:::

:::{grid-item-card}
:columns: 12 4 4 4

$\boldsymbol{r \to \infty}$
^^^

The converse of the previous case: $Δ q_{2^{-n}}(Φ)$ is infinitely larger than $Δ q_{2^{-n}}(Φ+2^{-n})$. We set $x_1 = 0$.

:::

:::{grid-item-card}
:columns: 12 12 12 12

$\boldsymbol{v \to 0}$
^^^

As $v$ vanishes, the distribution for $x_1$ approaches a Dirac delta centered on $\tfrac{1}{r+1}$.
In our implementation, we replace $x_1$ by a constant when $v < 10^{-8}$.

:::

:::{grid-item-card}
:columns: 12

$\boldsymbol{v \to \infty}$
^^^

Having $v$ go to infinity requires that $α$ and/or $β$ go to $0$ (see Eq. {eq}`eq_root-finding-problem` and {numref}`fig_polygamma`). The probability density of $x_1$ is then a Dirac delta: placed at $x_1=0$ if $α \to 0$, or placed at $x_1 = 1$ if $β \to 0$. If both $α$ and $β$ go to $0$, the PDF must be the sum of two weighted deltas:
```{math}
p(x_1) = w_0 δ(x_1 - 0) + w_1 δ(x_1 - 1) \,.
```
The weights $w_i$ can be determined by requiring that
```{math}
\EE[x_1] = r \,,
```
which yields
```{math}
\begin{aligned}
w_1 &= \frac{r}{r+1}\,, & w_2 &= \frac{1}{r+1} \,.
\end{aligned}
```
(For this special case, we revert to writing the condition in terms of a standard (Lebesgue) expectation, since the center (Eq. {eq}`eq_Aitchison-moments`) is undefined when $α, β \to 0$.)

Since we have already considered the special cases $r = 0$ and $r \to \infty$, we can assume $0 < r < \infty$. Then both $α$ and $β$ are zero, and $x_1$ should be a Bernoulli random variable with success probability $p = w_2 = \frac{1}{r+1}$.

:::

::::

```{code-cell}
:tags: [active-ipynb]

def get_beta_rv(r: Real, v: Real) -> Tuple[float]:
    """
    Return α and β corresponding to `r` and `v`.
    This function is not exported: it is used only for illustrative purposes
    within the notebook.
    """
    # Special cases for extreme values of r
    if r < 1e-12:
        return scipy.stats.bernoulli(0)  # Dirac delta at 0
    elif r > 1e12:
        return scipy.stats.bernoulli(1)  # Dirac delta at 1
    # Special cases for extreme values of v
    elif v < 1e-8:
        return get_beta_rv(r, 1e-8)
    elif v > 1e4:
        # (Actual draw function replaces beta by a Bernoulli in this case)
        return scipy.stats.bernoulli(1/(r+1))
    
    # if v < 1e-6:
    #     # Some unlucky values, like r=2.2715995006941436, v=6.278153793994013e-08,
    #     # seem to be particularly pathological for the root solver.
    #     # At least in the case above, the function is continuous at those values
    #     # (±ε returns very similar values for a and b).
    #     # Replacing these by nearby values which are more friendly to binary representation
    #     # seems to help.
    #     v = digitize(v, rtol=1e-5, show=False)
    
    # if 0.25 < r < 4:
    #     # Special case for r ≈ 1: improve initialization by first solving r=1 <=> α=β
    # Improve initialization by first solving r=1 <=> α=β
    x0 = scipy.optimize.brentq(f_mid, -5, 20, args=(v,))
    x0 = (x0, x0)
    # else:
    #     # Normal case: Initialize fit at (α, β) = (1, 1)
    #     x0 = (0, 0)
    res = scipy.optimize.root(f, x0, args=[math.log(r), v])
    if not res.success:
        logger.error("Failed to determine α & β parameters for beta distribution. "
                     f"Conditions were:\n  {r=}\n{v=}")
    α, β = np.exp(res.x)
    return scipy.stats.beta(α, β)
```

```{code-cell}
:tags: [remove-cell, active-ipynb]

# import jaxopt

# @partial(jax.jit, static_argnums=(0,))
# def jaxopt_bisection_solver(f, a, b, args):  # Signature is compatible with scipy.optimize.brentq
#     bisec = jaxopt.Bisection(f, -5, 20, check_bracket=False, jit=True)
#     return bisec.run(None, *args).params
    
# #@partial(jax.jit, static_argnames=("fun", "x0", "args", "method"))
# def jaxopt_mvroot_solver(fun, x0, args, method):  # Signature is compatible with scipy.optimize.root
#     rootsolver = jaxopt.ScipyRootFinding(method=method, optimality_fun=fun, jit=True)
#     res = rootsolver.run(x0, args)
#     return res.params, res.state.success
```

```{code-cell}
def scipy_mvroot_solver(fun, x0, args, method, root=scipy.optimize.root):
    res = root(fun, x0, args, method)
    return res.x, res.success
```

```{code-cell}
:tags: [hide-input]

def _draw_from_beta_scalar(r: Real, v: Real, rng: RNGenerator, n_samples: int=1,
                           *, _log=math.log, _exp=np.exp, _shape=np.shape,
                           bracketed_solver=scipy.optimize.brentq, mvroot_solver=scipy_mvroot_solver
                          ) -> Tuple[float]:
    rng = np.random.default_rng(rng)  # No-op if `rng` is already a Generator
    size = None if n_samples == 1 else (*_shape(r), n_samples)
    # Special cases for extreme values of r
    if r < 1e-12:
        special_val = 1           # EXIT AT END
    elif r > 1e12:
        special_val = 0           # EXIT AT END
    # Special cases for extreme values of v
    elif v < 1e-8:
        special_val = 1 / (1+r)   # EXIT AT END
    elif v > 1e4:
        # Replace beta by a Bernouilli distribution
        return rng.binomial(1, 1/(r+1), size=size)
    
    # Normal case
    else:

        # if v < 1e-6:
        #     # Some unlucky values, like r=2.2715995006941436, v=6.278153793994013e-08,
        #     # seem to be particularly pathological for the root solver.
        #     # At least in the case above, the function is continuous at those values
        #     # (±ε returns very similar values for a and b).
        #     # Replacing these by nearby values which are more friendly to binary representation
        #     # seems to help.
        #     v = digitize(v, rtol=1e-5, show=False)
        
        # if 0.25 < r < 4:
        #     # Special case for r ≈ 1 and 1 < v < 1e5: Initialize on the α=β line
        #     # In this case the initialization (0,0) is unstable, so we 
        # First find a better initialization by solving for the 1d case
        # where r=1 and therefore α=β.
        # (The limits where the normal case fails are around (r=1/3, v=1e4) and (r=3, v=1e4)
        # NB: The values -5 and 20 are slightly beyond the special case limits 5e-8 < v < 1e4 set above;
        #     since also the trigamma function is monotone, this should always find a solution.
        x0 = bracketed_solver(f_mid, -5, 20, args=(v,))
        
        x0 = (x0, x0)
        # else:
        #     # Normal case: Initialize fit at log(α, β) = (1, 1)
        #     x0 = (0., 0.)
        x, success = mvroot_solver(f, x0, args=[_log(r), v], method="hybr")
        if not success:
            logger.error("Failed to determine α & β parameters for beta distribution. "
                         f"Conditions were:\n  {r=}\n  {v=}")
        α, β = _exp(x)
        return rng.beta(α, β, size=size)
    
    # Finally, if `size` was passed, ensure result has the right shape
    # NB: We only reach this point if we go through one of the 3 first special cases
    if size:
        return np.array(special_val)[...,None].repeat(n_samples, axis=-1)
    else:
        return special_val

def draw_from_beta(r: Union[Real,Array[float,1]],
                   v: Union[Real,Array[float,1]],
                   rng: Optional[RNGenerator]=None,
                   n_samples: int=1
                  ) -> Tuple[float]:
    """
    Return α, β for a beta distribution with a metric variance `v` and center
    biased by `r`. More precisely, `r` is the ratio of the lengths ``c`` and
    ``1-c``, where ``c`` is the center.
    
    `r` and `v` may either be scalars or arrays
    """
    rng = np.random.default_rng(rng)  # No-op if `rng` is already a Generator
    
    if hasattr(r, "__iter__"):
        return np.array([_draw_from_beta_scalar(_r, _v, rng, n_samples)
                          for _r, _v in zip(r, v)])
    else:
        return _draw_from_beta_scalar(r, v, rng, n_samples)
```

```{code-cell}
:tags: [hide-input, active-ipynb, remove-cell]

# # We can’t jit because ScipyRootFinding.run (in the normal branch) is not jittable
# #@partial(jax.jit, static_argnames=("n_samples", "_log", "_exp", "_shape"))
# def _draw_from_beta_scalar_jax(r: Real, v: Real, rng: Array, n_samples: int=1,
#                                *, _log=jnp.log, _exp=jnp.exp, _shape=jnp.shape,
#                           ) -> Tuple[float]:
# #    rng = np.random.default_rng(rng)  # No-op if `rng` is already a Generator
#     size = None if n_samples == 1 else (*_shape(r), n_samples)
#     outshape = size or ()
#     rng, subkey = jax.random.split(rng)
#     # Special cases for extreme values of r
    
#     branch_idx = jnp.select(
#         [r == 0,      # special val 1 
#          r > 1e12,    # special val 0
#          v < 1e-8,    # special val 1 / (1+r)
#          v > 1e4  ],  # Replace beta by a Bernoulli distribution
#         jnp.arange(4),
#         default=4     # Normal branch
#     )
#     #x = jax.lax.switch(                              # What we actually want to use, but requires traceable _normal_branch (it automatically calls jit on its args)
#     x = (lambda i, flst, *args: flst[i](*args))(      # Workaround: does not automatically call jit, but also isn’t traceable (so vmap will break here)
#         branch_idx,
#         [ lambda r,v,size,subkey: jnp.array(1)      [...,None].repeat(n_samples, axis=-1).reshape(outshape),
#           lambda r,v,size,subkey: jnp.array(0)      [...,None].repeat(n_samples, axis=-1).reshape(outshape),
#           lambda r,v,size,subkey: jnp.array(1/(1+r))[...,None].repeat(n_samples, axis=-1).reshape(outshape),
#           lambda r,v,size,subkey: jax.random.bernoulli(subkey, 1/(r+1), shape=size).astype(float),
#           _normal_branch_beta                                                                ],
#         r, v, size, subkey  # operands
#     )
#     return rng, x

# def _normal_branch_beta(r, v, size, subkey,
#                         beta=jax.random.beta, _exp=jnp.exp, _log=jnp.log,
#                         bracketed_solver=jaxopt_bisection_solver,
#                         #mvroot_solver=jaxopt_mvroot_solver
#                         mvroot_solver=scipy_mvroot_solver
#                        ):
#     # First find a better initialization by solving for the 1d case
#     # where r=1 and therefore α=β.
#     # (The limits where the normal case fails are around (r=1/3, v=1e4) and (r=3, v=1e4)
#     # NB: The values -5 and 20 are slightly beyond the special case limits 5e-8 < v < 1e4 set above;
#     #     since also the trigamma function is monotone, this should always find a solution.
#     x0 = bracketed_solver(f_mid, -5, 20, args=(v,))
#     x0 = jnp.tile(x0, (2,))  # Convert x0 to (x0, x0)
#     x, success = mvroot_solver(f, x0, args=[_log(r), v], method="hybr")
#     try:
#         assert success
#     except AssertionError:
#         logger.error("Failed to determine α & β parameters for beta distribution. "
#                      f"Conditions were:\n  {r=}\n  {v=}")
#     α, β = _exp(x)
#     return beta(subkey, α, β, shape=size)
```

```{code-cell}
:tags: [active-ipynb, remove-cell]

# # Does not work: _draw_from_beta_scalar_jax needs to be traceable
# _draw_from_beta_scalar_jax_vectorized = jax.vmap(_draw_from_beta_scalar_jax, [0, 0, 0, None], 0)

# def draw_from_beta_jax(r: Union[Real,Array[float,1]],
#                        v: Union[Real,Array[float,1]],
#                        rng: Array[np.uint32],
#                        n_samples: int=1
#                       ) -> Tuple[float]:
#     """
#     Return α, β for a beta distribution with a metric variance `v` and center
#     biased by `r`. More precisely, `r` is the ratio of the lengths ``c`` and
#     ``1-c``, where ``c`` is the center.
    
#     `r` and `v` may either be scalars or arrays
#     """
    
    
#     if hasattr(r, "__iter__"):
#         rngkeys = jax.random.split(rng, len(r) + 1)
#         rng = rngkeys[0]
#         x = _draw_from_beta_scalar_jax_vectorized(r, v, rngkeys[1:], n_samples)
#     else:
#         rng, subkey = jax.random.split(rng)
#         x = _draw_from_beta_scalar_jax(r, v, subkey, n_samples)
#     return rng, x
```

```{code-cell}
:tags: [remove-cell, active-ipynb]

# # Does not work: _draw_from_beta_scalar_jax needs to be traceable
# draw_from_beta_jax(jnp.array([1., 2., 3.]), jnp.array([1., 2., 3.]), key, 4)
```

(supp_path-sampling_example-fitted-beta)=
### Examples of different fitted beta distributions

+++

Plotted below are the beta distributions for different values of $r$ and $v$.

```{code-cell}
:tags: [active-ipynb, hide-input, full-width]

%%opts Curve [title="Fitted beta distributions", ylim=(None,7)]
%%opts Table [title="Empirical statistics (4000 samples)"]
%%opts Layout [sublabel_format=""]

curves = {}
stats = {}
xarr = np.linspace(0, 1, 400)
for (r, v), c in zip(
            [(0.2, 1e-32), (0.2, 1e-16), (0.2, 1e-8), (0.2, 1e-4), (0.2, 1e-2), (0.2, 0.5),
             (0.5, 0.5), (0.5, 0.1), (0.5, 1),
             (1, 0.5), (1, 1e1), (1, 30), (1, 50), (1, 70), (1, 1e2), (1, 1e3), (1, 1e5), (1, 1e6),
             (5, 0.5), (5, 8), (5, 1e4), (5, 2e4), (5, 4e4), (5, 1e6), (5, 1e8), (5, 1e16), (5, 1e32),
             (6.24122778821756, 414.7130462762959),
             (2.2715995006941436, 6.278153793994013e-08),
             (2.271457193328191, 6.075242708902806e-08),
             (2.269182419251242, 6.794061846449025e-08),
             (2.691033486949275e-17, 0.02930210834055045)
            ],
        itertools.cycle(config.viz.colors.bright.cycle)):
    rv = get_beta_rv(r, v)
    if isinstance(rv.dist, scipy.stats.rv_discrete):
        # Dirac delta distribution
        p, = rv.args
        if p == 0:
            α, β = np.inf, 0
        elif p == 1:
            α, β = 0, np.inf
        else:
            α, β = 0, 0
    else:
        # rv is a beta random variable
        α, β = rv.args
    # α, β = get_beta_α_β(r, v)
    x = draw_from_beta(r, v, n_samples=4000)
    # rv = beta(α, β)
    # x = rv.rvs(4000)
    pdf = rv.pmf(xarr) if isinstance(rv.dist, scipy.stats.rv_discrete) else rv.pdf(xarr)
    curves[(r,v)] = hv.Curve(zip(xarr, pdf), label=f"{r=}, {v=}",
                             kdims=[hv.Dimension("x1", label="$x_1$")],
                             vdims=[hv.Dimension("px1", label="p($x_1$)")]  # non-TeX brackets to avoid legend issue 
                            ).opts(color=c)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", category=RuntimeWarning)
        stats[(r,v)] = tuple(f"{y:.3f}" for y in
                              (α, β, x.mean(), x.std(),
                               np.exp(digamma(α)) / (np.exp(digamma(α)) + np.exp(digamma(β))),
                               0.5 * (polygamma(1, α) + polygamma(1, β))
                              ))

hmap = hv.HoloMap(curves, kdims=["r", "v"])
dists = hmap.overlay()
dists.opts(legend_position="right")
dists.opts(width=500, backend="bokeh")
dists.opts(fig_inches=7, aspect=2.5, legend_cols=2, backend="matplotlib")
```

+++ {"tags": ["remove-cell", "skip-execution"]}

Extra test, for a point that is especially sensitive to numerical issues: despite the very tight distribution, only 10-25% points raise a warning.

```{code-cell}
:tags: [remove-cell, skip-execution, active-ipynb]

r=2.2691824192512438;    Δr = 0.000000000000001
v=6.79406184644904e-08;  Δv = 1e-22
s=2
rng = np.random.default_rng(45)
for r,v in rng.uniform([r-s*Δr, v-s*Δv], [r+s*Δr, v+s*Δv], size=(40,2)):
    draw_from_beta(r,v)
```

Statistics for the fitted beta distributions. $\mathbb{E}[x_1]$ and $\mathrm{std}[x_1]$ are computed from 4000 samples. $\mathbb{E}_a[x_1]$ and $\mathrm{Mvar}[x_1,x_2]$ are computed using the expressions above.

```{code-cell}
:tags: [active-ipynb, hide-input, full-width]

def clean_table_mpl(plot, element):
    "TODO: Modify table to remove vertical lines"
    table = plot.handles["artist"]

flat_stats = [k + v for k,v in stats.items()]
dim_Ex1 = hv.Dimension("Ex1", label="$\mathbb{E}[x_1]$")
dim_stdx1 = hv.Dimension("stdx1", label="$\mathrm{std}[x_1]$")
dim_Eax1 = hv.Dimension("Eax1", label="$\mathbb{E}_a[x_1]$")
dim_Mvar = hv.Dimension("Mvar", label="$\mathrm{Mvar}[x_1,x_2]$")
stattable = hv.Table(flat_stats, kdims=["r", "v"],
                     vdims=["α", "β", dim_Ex1, dim_stdx1, dim_Eax1, dim_Mvar])
# We need to increase max_value_len because Holoviews uses the unformatted
# length to decide when to truncate
stattable.opts(max_rows=len(stattable)+1)  # Ensure all rows are shown
stattable.opts(fig_inches=18, aspect=2.5, max_value_len=30, hooks=[clean_table_mpl], backend="matplotlib")
```

+++ {"tags": ["remove-cell"]}

`draw_from_beta` also supports passing `r` and `v` as vectors. This is mostly a convenience: internally the vectors are unpacked and $(α,β)$ are solved for individually.

```{code-cell}
:tags: [active-ipynb, skip-execution, remove-cell, full-width]

r_vals, v_vals = np.array(
    [(0.2, 1e-32), (0.2, 1e-16), (0.2, 1e-8), (0.2, 1e-4), (0.2, 1e-2), (0.2, 0.5),
     (0.5, 0.5), (0.5, 0.1), (0.5, 1),
     (1, 0.5), (1, 1e1), (1, 1e2), (1, 1e3), (1, 1e5), (1, 1e6),
     (5, 0.5), (5, 8), (5, 1e4), (5, 2e4), (5, 4e4), (5, 1e6), (5, 1e8), (5, 1e16), (5, 1e32)]
).T

# α, β = get_α_β(r_vals, v_vals)
# rv = beta(α, β)
# x = rv.rvs((4000, len(r_vals)))
x = draw_from_beta(r_vals, v_vals, n_samples=4000)
flat_stats = np.stack((r_vals, v_vals, x.mean(axis=-1), x.std(axis=-1)
                       #np.exp(digamma(α)) / (np.exp(digamma(α)) + np.exp(digamma(β))),
                       #0.5 * (polygamma(1, α) + polygamma(1, β))
                      )).T
stattable = hv.Table(flat_stats, kdims=["r", "v"],
                     vdims=[dim_Ex1, dim_stdx1])
stattable.opts(max_rows=len(stattable)+1)
stattable.opts(fig_inches=14, aspect=1.8, max_value_len=30, hooks=[clean_table_mpl], backend="matplotlib")
```

+++ {"tags": ["remove-cell"]}

### Comparative timings NumPy vs JAX

:::{admonition} Takeaway
:class: important

- JAX random number generation is about 50% slower than NumPy.
- We can JIT `f` and `f_mid`, because JAX provides `digamma` and `polygamma`.
  However, this does not affect runtime in a meaningful way.
- We can write `_draw_from_beta_scalar` as a JAX function, but without JITing, it is generally a few fold slower than the NumPy version (this is expected, because JAX functions have higher overhead).
- If we rewrite it in a functional style, as would be required for JITing, it is *many* fold slower.
  This is most likely due to two things (further profiling would be required to determine their relative importance):
  + the extra function overhead, which would disappear if we could actually JIT;
  + the use of the `Bisection` function instead of `brentq`
- We can use `jaxopt.Bisection` to get a jittable root finding function. However, it is possibly slower than `brentq`.
- The `jaxopt` library also provides `ScipyRootFinding`, but this just wraps `scipy.optimize.root`. So it allows the arguments to be jitted, and automatically differentiated, but it is not itself jittable.
- Consequently, `_draw_from_beta_scalar` is also not jittable.
- As a further consequence, we also can’t use `vmap` to vectorize `_draw_from_beta`, since it needs to trace through the function.
- **Current verdict** As it stands, JAX adds complexity with no gain. To make it worthwhile, one would need either need to at least implement a jittable multivariate solver, and then see if by jitting the whole function we obtain meaningful improvement.
  + `jaxopt` is written using [`lax.custom_root`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.custom_root.html#jax.lax.custom_root). Possibly this could also be useful.
  + Alternatively, perhaps one can rework the problem so that finding the $α, β$ pair does not require solving a 2-d optimization problem, but some other type of problem for which `jaxopt` has a jittable function.
:::

```{code-cell}
:tags: [remove-cell, active-ipynb]

# %timeit jax.random.split(subkey)
# %timeit jax.random.beta(subkey, 1, 1)

# %timeit rng.beta(1, 1)
```

:::{list-table} **Timing `_draw_from_beta_scalar`.** Timings marked *"JAX (imperative)"* use an implementation very near the NumPy one, with just minimal changes (replacement of np with jnp, RNG keys, etc.). In particular, the imperative version still uses `brentq`. Timings marked *"JAX (functional)"* use the more substantial rewrite still present above, which if in theory could be compiled if the multivariate root solver (`jaxopt_mvroot_solver`) was jittable. Using `scipy_mvroot_solver` instead results in very similar times.
:header-rows: 2

* -
  -
  -
  - No JIT
  -
  - JIT-ed *gamma
  -
  - JIT-ed bisec
* - L
  - r
  - v
  - NumPy
  - JAX (imperative)
  - NumPy
  - JAX (imperative)
  - JAX (functional)
* - 1
  - 0.2
  - 1e-32
  - 1.14 μs ± 16.2 ns (±2%)
  - 508 ns ± 0.614 ns (±2%)
  - 1.1 μs ± 0.677 ns (±4%)
  - 501 ns ± 0.388 ns (±1%)
  - 1.57 ms ± 59.6 µs
* - 1
  - 0.2
  - 1e-16
  - 1.11 μs ± 2.95 ns (±2%)
  - 507 ns ± 0.56 ns (±0%)
  - 1.1 μs ± 5.78 ns (±1%)
  - 501 ns ± 0.651 ns (±0%)
  - 1.55 ms ± 13.9 µs
* - 1
  - 0.2
  - 1e-08
  - 550 μs ± 6.47 μs (±1%)
  - 2.52 ms ± 12.4 μs (±1%)
  - 560 μs ± 782 ns (±3%)
  - 2.07 ms ± 16.8 μs (±5%)
  - 62.3 ms ± 489 µs
* - 1
  - 0.2
  - 0.0001
  - 717 μs ± 12 μs (±1%)
  - 2.84 ms ± 7.44 μs (±3%)
  - 698 μs ± 2.97 μs (±1%)
  - 2.22 ms ± 17.4 μs (±2%)
  - 75.6 ms ± 556 µs
* - 1
  - 0.2
  - 0.01
  - 737 μs ± 8.1 μs (±1%)
  - 2.91 ms ± 8.94 μs (±1%)
  - 735 μs ± 5.79 μs (±3%)
  - 2.23 ms ± 12.4 μs (±3%)
  - 76.5 ms ± 1.42 ms
* - 1
  - 0.2
  - 0.5
  - 804 μs ± 6.82 μs (±1%)
  - 3.02 ms ± 16.4 μs (±1%)
  - 805 μs ± 16.5 μs (±2%)
  - 2.31 ms ± 19.4 μs (±1%)
  - 83.3 ms ± 1.88 ms
* - 1
  - 0.5
  - 0.5
  - 757 μs ± 8.25 μs (±1%)
  - 2.88 ms ± 10.5 μs (±7%)
  - 764 μs ± 857 ns (±1%)
  - 2.26 ms ± 12 μs (±1%)
  - 76.1 ms ± 886 µs
* - 1
  - 0.5
  - 0.1
  - 735 μs ± 10.2 μs (±2%)
  - 2.86 ms ± 5.1 μs (±2%)
  - 754 μs ± 3.06 μs (±2%)
  - 2.28 ms ± 42 μs (±3%)
  - 75.7 ms ± 496 µs
* - 1
  - 0.5
  - 1.0
  - 736 μs ± 8.01 μs (±1%)
  - 2.86 ms ± 11.1 μs (±3%)
  - 727 μs ± 3.3 μs (±4%)
  - 2.26 ms ± 21.9 μs (±0%)
  - 75.6 ms ± 406 µs 
* - 1
  - 1.0
  - 0.5
  - 512 μs ± 6.09 μs (±2%)
  - 2.36 ms ± 9 μs (±2%)
  - 559 μs ± 2 μs (±2%)
  - 2.06 ms ± 13 μs (±2%)
  - 49.2 ms ± 517 µs
* - 1
  - 1.0
  - 10.0
  - 516 μs ± 7.56 μs (±2%)
  - 2.36 ms ± 7.87 μs (±2%)
  - 557 μs ± 2.22 μs (±3%)
  - 2.05 ms ± 19.1 μs (±2%)
  - 49.1 ms ± 301 µs
* - 1
  - 1.0
  - 100.0
  - 479 μs ± 7.3 μs (±1%)
  - 2.36 ms ± 15.4 μs (±1%)
  - 498 μs ± 1.66 μs (±2%)
  - 1.99 ms ± 17.1 μs (±1%)
  - 48.9 ms ± 252 µs
* - 1
  - 1.0
  - 1000.0
  - 461 μs ± 10.3 μs (±2%)
  - 2.31 ms ± 2.92 μs (±2%)
  - 483 μs ± 3.23 μs (±1%)
  - 1.98 ms ± 8.52 μs (±1%)
  - 49 ms ± 312 µs
* - 1
  - 1.0
  - 100000.0
  - 2.11 μs ± 62.1 ns (±8%)
  - 810 μs ± 1.51 μs (±1%)
  - 2.07 μs ± 33.3 ns (±4%)
  - 810 μs ± 2.65 μs (±2%)
  - 1.29 ms ± 19.6 µs
* - 1
  - 1.0
  - 1000000.0
  - 2.07 μs ± 18.5 ns (±4%)
  - 811 μs ± 2.22 μs (±1%)
  - 2.05 μs ± 15.6 ns (±3%)
  - 809 μs ± 1.35 μs (±0%)
  - 1.28 ms ± 5.65 µs
* - 1
  - 5.0
  - 0.5
  - 805 μs ± 10 μs (±1%)
  - 3.02 ms ± 4.95 μs (±2%)
  - 790 μs ± 628 ns (±5%)
  - 2.27 ms ± 8.23 μs (±2%)
  - 85.4 ms ± 2.6 ms
* - 1
  - 5.0
  - 8.0
  - 853 μs ± 11.5 μs (±0%)
  - 3.16 ms ± 34.4 μs (±1%)
  - 829 μs ± 3.2 μs (±2%)
  - 2.31 ms ± 17.3 μs (±1%)
  - 90.6 ms ± 1.18 ms
* - 1
  - 5.0
  - 10000.0
  - 587 μs ± 9.9 μs (±3%)
  - 2.67 ms ± 28.2 μs (±0%)
  - 573 μs ± 990 ns (±1%)
  - 2.07 ms ± 18.3 μs (±1%)
  - 62 ms ± 490 µs
* - 1
  - 5.0
  - 20000.0
  - 2.06 μs ± 19.8 ns (±2%)
  - 813 μs ± 5.46 μs (±2%)
  - 2.05 μs ± 16.6 ns (±1%)
  - 808 μs ± 1.38 μs (±2%)
  - 1.28 ms ± 6 µs
* - 1
  - 5.0
  - 40000.0
  - 2.16 μs ± 10.2 ns (±3%)
  - 819 μs ± 7.11 μs (±1%)
  - 2.07 μs ± 9.88 ns (±3%)
  - 806 μs ± 1.95 μs (±0%)
  - 1.28 ms ± 11.1 µs
* - 1
  - 5.0
  - 1000000.0
  - 2.06 μs ± 13.3 ns (±3%)
  - 834 μs ± 1.26 μs (±2%)
  - 2.06 μs ± 15.5 ns (±2%)
  - 814 μs ± 2.21 μs (±1%)
  - 1.28 ms ± 13.6 µs
* - 1
  - 5.0
  - 100000000.0
  - 2.08 μs ± 13 ns (±3%)
  - 835 μs ± 1.88 μs (±2%)
  - 2.05 μs ± 12 ns (±2%)
  - 813 μs ± 1.25 μs (±1%)
  - 1.27 ms ± 10.6 µs
* - 1
  - 5.0
  - 1e+16
  - 2.06 μs ± 24 ns (±3%)
  - 827 μs ± 2.65 μs (±5%)
  - 2.05 μs ± 15.4 ns (±4%)
  - 812 μs ± 2.74 μs (±0%)
  - 1.27 ms ± 5.78 µs
* - 1
  - 5.0
  - 1e+32
  - 2.11 μs ± 38.9 ns (±2%)
  - 822 μs ± 9.73 μs (±1%)
  - 2.06 μs ± 13.8 ns (±6%)
  - 807 μs ± 2.03 μs (±1%)
  - 1.28 ms ± 33.4 µs 
:::

```{code-cell}
:tags: [active-ipynb, remove-cell]

# r_vals, v_vals = np.array(
#     [(0.2, 1e-32), (0.2, 1e-16), (0.2, 1e-8), (0.2, 1e-4), (0.2, 1e-2), (0.2, 0.5),
#      (0.5, 0.5), (0.5, 0.1), (0.5, 1),
#      (1, 0.5), (1, 1e1), (1, 1e2), (1, 1e3), (1, 1e5), (1, 1e6),
#      (5, 0.5), (5, 8), (5, 1e4), (5, 2e4), (5, 4e4), (5, 1e6), (5, 1e8), (5, 1e16), (5, 1e32)]
# ).T
```

```{code-cell}
:tags: [active-ipynb, remove-cell]

# key = jax.random.PRNGKey(0)
# key, *subkeys = jax.random.split(key, 2)
# rng = np.random.Generator(np.random.PCG64(0))

# from collections import namedtuple
# ResData = namedtuple("ResData", ["avg", "std", "diff_percent"])
# def get_resdata(res_lst):
#     avgs = [res.average for res in res_lst]
#     stds = [res.stdev for res in res_lst]
#     slst = []
#     i = np.argmin(avgs)
#     diff_percent = (np.max(avgs) - np.min(avgs)) / avgs[i] if len(res_lst) > 1 \
#                    else None
#     return ResData(avgs[i], stds[i], diff_percent)
```

```{code-cell}
:tags: [remove-cell, active-ipynb]

# try:
#     time_data = np.load("timing_cache_jax-vs-numpy_draw-from-beta.npy")
# except FileNotFoundError:
#     time_results = {}
# else:
#     time_results = {(L, r, v): (ResData(npa, npb, npc), ResData(jaxa, jaxb, jaxc))
#                     for (L, r, v, npa, npb, npc, jaxa, jaxb, jaxc) in time_data}
```

```{code-cell}
:tags: [active-ipynb, remove-cell]

# def timing_run(func, desc, rng_lst, time_results):
#     progL = tqdm([1], desc="Sample size")  # [1, 7, 49, 343]
#     progrv = tqdm(list(zip(r_vals, v_vals)), desc="r, v")
#     for L in progL:
#         progrv.reset()
#         for r, v in progrv:
#             if (L, r, v, desc) in time_results:
#                 continue
#             # Warm-up to make sure compilation is not included in profiling time
#             _, x = func(r, v, rng_lst[0]); getattr(x, "block_until_ready", lambda:None)()
#             res_lst = []
#             for rng in rng_lst:
#                 res = %timeit -o func(r, v, rng)
#                 res_lst.append(res)
#             time_results[(L, r, v, desc)] = get_resdata(res_lst)
```

```{code-cell}
:tags: [active-ipynb, remove-cell, skip-execution]

# timing_run(_draw_from_beta_scalar_jax, "jax, functional", subkeys, time_results=time_results)
```

```{code-cell}
:tags: [active-ipynb, remove-cell]

# progL = tqdm([1], desc="Sample size")  # [1, 7, 49, 343]
# progrv = tqdm(list(zip(r_vals, v_vals)), desc="r, v")
# for L in progL:
#     progrv.reset()
#     for r, v in progrv:
#         if (L, r, v) in time_results:
#             continue
#         res_np = []
#         res_jax = []
#         for subkey in subkeys:
#             res = %timeit -o _draw_from_beta_scalar(r, v, rng)
#             res_np.append(res)
#             res = %timeit -o _draw_from_beta_scalar_jax(r, v, subkey)
#             res_jax.append(res)
#         time_results[(L, r, v)] = (get_resdata(res_np), get_resdata(res_jax))
```

```{code-cell}
:tags: [remove-cell, active-ipynb]

# _time_data = np.array([
#     [L, r, v, *res_np, *res_jax]
#     for (L, r, v), (res_np, res_jax) in time_results.items()
# ])
```

```{code-cell}
:tags: [active-ipynb, remove-cell]

# np.save("timing_cache_jax-vs-numpy_draw-from-beta", _time_data)
```

```{code-cell}
:tags: [remove-cell, active-ipynb]

# for ((L, r, v, npa, npb, npc, jaxa, jaxb, jaxc),
#      (_, _, _, _npa, _npb, _npc, _jaxa, _jaxb, _jaxc)) in zip(time_data, _time_data):
#     print("* -", int(L))
#     print("  -", r)
#     print("  -", v)
#     print("  -", time_str(ResData(npa, npb, npc)))
#     print("  -", time_str(ResData(jaxa, jaxb, jaxc)))
#     print("  -", time_str(ResData(_npa, _npb, _npc)))
#     print("  -", time_str(ResData(_jaxa, _jaxb, _jaxc)))                       
```

```{code-cell}
:tags: [remove-cell, active-ipynb]

# def format_with_unit(val, unit):
#     if val >= 1:
#         pass
#     elif val >= 1e-3:
#         val, unit = val*1e3, f"m{unit}"
#     elif val >= 1e-6:
#         val, unit = val*1e6, f"μ{unit}"
#     else:
#         val, unit = val*1e9, f"n{unit}"
#     return f"{val:.3g} {unit}"
# def time_str(data: ResData):
#     return f"{format_with_unit(data.avg, 's')} ± {format_with_unit(data.std, 's')} (±{data.diff_percent*100:.0f}%)"
# time_table = hv.Table([(int(L), r, v, time_str(ResData(npa, npb, npc)), time_str(ResData(jaxa, jaxb, jaxc)))
#                        for (L, r, v, npa, npb, npc, jaxa, jaxb, jaxc) in time_data],
#                       kdims=["# fits", "r", "v"], vdims=["NumPy", "JAX"])
# time_table.opts(aspect=20/len(time_table), fig_inches=10, backend="matplotlib") \
#           .opts(fit_columns=True, width=1000, backend="bokeh", )
```

+++ {"tags": ["remove-cell"]}

### Timings for the root solver

+++ {"tags": ["remove-cell"]}

Reported below are timings for different numbers of fits, comparing a “loop” approach where `get_α_β` is called for each pair `(r, v)`, and a “vectorized” approach where `r` and `v` are passed as vectors.

At the moment there is no clear benefit to using the vectorized form; this is likely because it performs the fit in a much higher dimensional space, and it must continue calculations with this large vector until all equations are solved.

NB: Timings were done for an older version, where the function returned the $α, β$ parameters rather than a beta random variable. This function also performed vectorized operations by enlarging the fit dimension, rather than the current approach of looping over $(r, v)$ pairs. The observation that this approach was in general no faster than looping partly motivated the change, so we keep these timings results as documentation.

+++ {"tags": ["remove-cell"]}

```python
time_results = []
for L in tqdm([1, 7, 49, 343]):
    r_vals = np.random.uniform(low=0, high=1, size=L)
    v_vals = np.random.exponential(3, size=L)
    [get_α_β(r, v) for r, v in zip(r_vals, v_vals)]
    res_loop = %timeit -q -o [get_α_β(r, v) for r, v in zip(r_vals, v_vals)]
    res_vec = %timeit -q -o get_α_β(r_vals, v_vals)
    time_results.append((L, res_loop, res_vec))

def time_str(time_res): s = str(time_res); return s[:s.index(" per loop")]
time_table = hv.Table([(L, time_str(res_loop), time_str(res_vec))
                       for L, res_loop, res_vec in time_results],
                      kdims=["# fits"], vdims=["loop", "vectorized"])
time_table.opts(aspect=4, fig_inches=7)
```

+++ {"tags": ["remove-cell"]}

| # fits |             loop |        vectorized |
|-------:|-----------------:|------------------:|
|      1 | 555 μs ± 14.2 μs |  515 μs ± 20.2 μs |
|      7 | 4.06 ms ± 108 μs |    1.8 ms ± 20 μs |
|     49 | 28.5 ms ± 589 μs |  17.1 ms ± 146 μs |
|    343 | 187 ms ± 2.15 ms | 3.95 ms ± 33.1 ms |

+++ {"tags": ["remove-cell"]}

The test above samples $r$ from the entire interval $[0, 1]$, but we get similar results when restricting values to the “easy” region $[0.4, 0.6]$. Reducing the values of $v$ (by sampling from a distribution with lighter tail) does bring down the execution time of the vectorized approach. This is consistent with the hypothesis that a few especially difficult $(r,v)$ combinations are slowing down the computations.

+++ {"tags": ["remove-cell"]}

```python
time_results2 = []
for L in tqdm([1, 7, 49, 343]):
    r_vals = np.random.uniform(low=0.4, high=.6, size=L)
    v_vals = np.random.exponential(1, size=L)
    [get_α_β(r, v) for r, v in zip(r_vals, v_vals)]
    res_loop = %timeit -q -o [get_α_β(r, v) for r, v in zip(r_vals, v_vals)]
    res_vec = %timeit -q -o get_α_β(r_vals, v_vals)
    time_results2.append((L, res_loop, res_vec))

def time_str(time_res): s = str(time_res); return s[:s.index(" per loop")]
time_table = hv.Table([(L, time_str(res_loop), time_str(res_vec))
                       for L, res_loop, res_vec in time_results2],
                      kdims=["# fits"], vdims=["loop", "vectorized"])
time_table.opts(aspect=4, fig_inches=7)
```

+++ {"tags": ["remove-cell"], "editable": true, "slideshow": {"slide_type": ""}}

| # fits |              loop |        vectorized |
|-------:|------------------:|------------------:|
|      1 |  474 μs ± 12.3 μs |  452 μs ± 2.62 μs |
|      7 | 3.92 ms ± 43.7 μs | 1.65 ms ± 22.9 μs |
|     49 |  29.2 ms ± 167 μs | 16.6 ms ± 84.8 μs |
|    343 |  214 ms ± 4.27 ms | 1.37 ms ± 6.24 ms |

+++

(supp_path-sampling_implementation)=
## Implementation

+++

### Generate a single path

Now that we know how to construct an sampling distribution for the increments, sampling an entire path is just a matter of repeating the process recursively until we reach the desired resolution.

```{code-cell}
def generate_path_hierarchical_beta(
        qstar: Callable, deltaEMD: Callable, c: float,
        qstart: float, qend: float, res: int=8, rng=None,
        *, Phistart: float=0., Phiend: float=1.
    ) -> Tuple[Array[float,1], Array[float,1]]:
    """
    Returned path has length ``2**res + 1``.
    If `qstar` and`Mvar` have a different length, they are linearly-
    interpolated to align with the returned array `Φhat`.
    
    Parameters
    ----------
    qstar: Empirical (mixed) quantile function. The generated path will
       be centered on this trace. Although any callable mapping [0, 1] to R
       is accepted, in all practical use cases this will be the output of
       :func:`emd.make_empirical_risk_ppf`.
    deltaEMD: Callable returning discrepancy values (i.e. {math}`δ^\mathrm{EMD}`).
       Like `qstar`, this must map [0, 1] to R+.
    c: Proportionality constant relating δEMD to
       the square root of the metric variance.
    qstart, qend: The end point ordinates. The hierarchical beta process "fills in"
       the space between a start and an end point, but it needs the value of _q_
       to be given at those points. These end points are by default Φ=0 and Φ=1,
       which correspond to the bounds of a quantile function.
    res: Returned paths have length ``2**res+1``, and therefore ``2**res`` increments.
       Typical values of `res` are 6, 7 and 8, corresponding to paths of length
       64, 128 and 256. Smaller may be useful to accelerate debugging. Larger values
       increase the computation cost with typically negligible improvements in accuracy.
       Must be at least 1.
    rng: Any argument accepted by `numpy.random.default_rng` to initialize an RNG.
    Phistart, Phiend: (Keyword only) The abscissa corresponding to the ordinates
       `qstart` and `qend`. The default values are 0 and 1, appropriate for generating
       a quantile function.
    
    Returns
    -------
    The pair Φhat, qhat.
        Φhat: Array of equally spaced values between 0 and 1, with step size ``2**-res``.
        qhat: The generated path, evaluated at the values listed in `Φhat`.
    """
    # Validation
    res = int(res)  # sourcery skip: remove-unnecessary-cast
    if Phistart >= Phiend:
        raise ValueError("`Phistart` must be strictly smaller than `Phiend`. "
                         f"Received:\n  {Phistart=}\n  {Phiend=}")
    # if not (len(Phi) == len(qstar) == len(Mvar)):
    #     raise ValueError("`Phi`, `qstar` and `Mvar` must all have "
    #                      "the same shape. Values received have the respective shapes "
    #                      f"{np.shape(Phi)}, {np.shape(qstar)}, {np.shape(Mvar)}")
    if res < 1:
        raise ValueError("`res` must be greater or equal to 1.")
    rng = np.random.default_rng(rng)  # No-op if `rng` is already a Generator
    # Interpolation
    N  = 2**res + 1
    Φarr = np.linspace(Phistart, Phiend, N)
    qsarr = qstar(Φarr)
    # Pre-computations
    Mvar = c * deltaEMD(Φarr)**2
    # Algorithm
    qhat = np.empty(N)
    qhat[0] = qstart
    qhat[-1] = qend
    for n in range(1, res+1):
        Δi = 2**(res-n)
        i = np.arange(Δi, N, 2*Δi)  # Indices for the new values to insert
        d = qhat[i+Δi] - qhat[i-Δi] # Each pair of increments must sum to `d`
        r = (qsarr[i] - qsarr[i-Δi]) / (qsarr[i+Δi]-qsarr[i])  # Ratio of first/second increments
        v = 2*Mvar[i]
        # Prevent failure in pathological cases where `q` is flat in some places
        if ((qsarr[i+Δi] - qsarr[i-Δi]) == 0).any():   # If both increments are zero, then the calculation for `r` gives 0/0 => NaN
            logger.warning("The quantile function is not strictly increasing. Non-increasing intervals were skipped. The sampling of EMD paths may be less accurate.")
            keep = ((qsarr[i+Δi] - qsarr[i-Δi]) > 0)  # `draw_from_beta` can deal with `r=0` and `r=inf`, but not `r=NaN`.
            x1 = np.zeros_like(r)
            x1[keep] = draw_from_beta(r[keep], v[keep], rng=rng)
        else:
            x1 = draw_from_beta(r, v, rng=rng)
        qhat[i] = qhat[i-Δi] + d * x1
    return Φarr, qhat
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
qstar = np.log
σtilde = lambda Φ: 1.5*np.ones_like(Φ)
Φhat, qhat = generate_path_hierarchical_beta(
    qstar, σtilde, c=1, qstart=np.log(0.01), qend=np.log(1),
    res=8, rng=None, Phistart=0.01)  # Can’t start at zero precisely because log(0) is undefined

Φarr = np.linspace(0.01, 1, 20)
curve_qstar = hv.Curve(zip(Φarr, qstar(Φarr)), label=r"$\tilde{l}$", kdims=["Φ"])
curve_qhat = hv.Curve(zip(Φhat, qhat), label=r"$\hat{l}$", kdims=["Φ"])
fig.opts(ylabel="")
```

### Generate ensemble of sample paths

:::{Note} `generate_quantile_paths` and `generate_path_hierarchical_beta` are the only two public functions exposed by this module.
:::

To generate $R$ paths, we repeat the following $R$ times:
1. Select start and end points by sampling $\nN(\tilde{Φ}[0], \lnLtt{}[0])$ and $\nN(\tilde{Φ}[-1], \lnLtt{}[-1])$.
2. Call `generate_path_hierarchical_beta`.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
def generate_quantile_paths(qstar: Callable, deltaEMD: Callable, c: float,
                            M: int, res: int=8, rng=None,
                            *, Phistart: float=0., Phiend: float=1,
                            progbar: Union[Literal["auto"],None,tqdm,"mp.queues.Queue"]="auto",
                            previous_M: int=0
                           ) -> Generator[Tuple[Array[float,1], Array[float,1]], None, None]:
    r"""
    Generate `M` distinct quantile paths, with trajectory and variability determined
    by `qstar` and `deltaEMD`.
    Paths are generated using the hierarchical beta algorithm, with normal distributions
    for the end points and beta distributions for the increments.
    
    Returned paths have length ``2**res + 1``.
    Typical values of `res` are 6, 7 and 8, corresponding to paths of length
    64, 128 and 256. Smaller values may be useful to accelerate debugging. Larger values
    increase the computation cost with (typically) negligible improvements in accuracy.
    
    .. Note:: When using multiprocessing to call this function multiple times,
       use either a `multiprocessing.Queue` or `None` for the `progbar` argument.
    
    Parameters
    ----------
    qstar: Empirical (mixed) quantile function. The generated paths will
       be centered on this trace. Although any callable mapping [0, 1] to R
       is accepted, in all practical use cases this will be the output of
       :func:`emd.make_empirical_risk_ppf`.
    deltaEMD: Callable returning scaled discrepancy values
       (i.e. {math}`δ^\mathrm{EMD}`). Like `qstar`, this must map [0, 1] to R+.
    c: Proportionality constant relating δEMD to
       the square root of the metric variance.
    M: Number of paths to generate.
    res: Returned paths have length ``2**res + 1``.
       Typical values of `res` are 6, 7 and 8, corresponding to paths of length
       64, 128 and 256. Smaller may be useful to accelerate debugging, but larger
       values are unlikely to be useful.
    rng: Any argument accepted by `numpy.random.default_rng` to initialize an RNG.
    progbar: Control whether to create a progress bar or use an existing one.
       - With the default value 'auto', a new tqdm progress is created.
         This is convenient, however it can lead to many bars being created &
         destroyed if this function is called within a loop.
       - To prevent this, a tqdm progress bar can be created externally (e.g. with
         ``tqdm(desc="Generating paths")``) and passed as argument.
         Its counter will be reset to zero, and its set total to `M` + `previous_M`.
       - (Multiprocessing): To support updating progress bars within child processes,
         a `multiprocessing.Queue` object can be passed, in which case no
         progress bar is created or updated. Instead, each time a quantile path
         is sampled, a value is added to the queue with ``put``. This way, the
         parent process can update a progress by consuming the queue; e.g.
         ``while not q.empty(): progbar.update()``.
         The value added to the queue is `M`+`previous_M`, which can be
         used to update the total value of the progress bar.
       - A value of `None` prevents displaying any progress bar.

    previous_M: Used only to improve the display of the progress bar:
       the indicated total on the progress bar will be `M` + `previous_M`.
       Use this when adding paths to a preexisting ensemble.

    Yields
    ------
    Tuples of two 1-d arrays: (Φhat, qhat).
    
    Notes
    -----
    Returned paths always have an odd number of steps, which as a side benefit is
    beneficial for integration with Simpson's rule.
    """
    rng = np.random.default_rng(rng)
    total = M + previous_M
    progbar_is_queue = ("multiprocessing.queues.Queue" in str(type(progbar).mro()))  # Not using `isinstance` avoids having to import multiprocessing & multiprocessing.queues
    if isinstance(progbar, str) and progbar == "auto":
        progbar = tqdm(desc="Sampling quantile paths", leave=False,
                       total=total)
    elif progbar is not None and not progbar_is_queue:
        progbar.reset(total)
        if previous_M:
            # Dynamic miniters don’t work well with a restarted prog bar: use whatever miniter was determined on the first run (or 1)
            progbar.miniters = max(progbar.miniters, 1)
            progbar.dynamic_miniters = False
            progbar.n = previous_M
            progbar.refresh()
    for r in range(M):  # sourcery skip: for-index-underscore
        for _ in range(100):  # In practice, this should almost always work on the first try; 100 failures would mean a really pathological probability
            qstart  = rng.normal(qstar(Phistart) , math.sqrt(c)*deltaEMD(Phistart))
            qend = rng.normal(qstar(Phiend), math.sqrt(c)*deltaEMD(Phiend))
            if qstart < qend:
                break
        else:
            raise RuntimeError("Unable to generate start and end points such that "
                               "start < end. Are you sure `qstar` is compatible "
                               "with monotone paths ?")
        Φhat, qhat = generate_path_hierarchical_beta(
            qstar, deltaEMD, c, qstart=qstart, qend=qend,
            res=res, rng=rng, Phistart=Phistart, Phiend=Phiend)
        
        yield Φhat, qhat
        
        if progbar_is_queue:
            progbar.put(total)
        elif progbar is not None:
            progbar.update()
            time.sleep(0.05)  # Without a small wait, the progbar might not update
```

### Usage example

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
qstar = np.log  # Dummy function; normally this would be obtain from data
σtilde = lambda Φ: 1.5*np.ones_like(Φ)

colors = cycle = config.viz.colors.bright

curves_qhat = []
for Φhat, qhat in generate_quantile_paths(
        qstar, σtilde, c=1, M=10, res=8, rng=None, Phistart=0.01):
    curves_qhat.append(hv.Curve(zip(Φhat, qhat), label=r"$\hat{l}$", kdims=["Φ"])
                       .opts(color=colors.grey))

Φtilde = np.linspace(0.01, 1, 20)
curve_qstar = hv.Curve(zip(Φtilde, qstar(Φtilde)), label=r"$\tilde{l}$", kdims=["Φ"]) \
                        .opts(color=colors.blue)

hv.Overlay((*curves_qhat, curve_qstar)).opts(ylabel="")
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input, active-ipynb]
---
GitSHA()
```
