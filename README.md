# [JaxLinOp](https://github.com/JaxGaussianProcesses/JaxLinOp)

[![PyPI version](https://badge.fury.io/py/JaxLinOP.svg)](https://badge.fury.io/py/JaxLinOP)

`JaxLinOp` is a lightweight linear operator library written in [`jax`](https://github.com/google/jax).

# Overview
Consider solving a diagonal matrix $A$ against a vector $b$.

```python
import jax.numpy as jnp

n = 1000
diag = jnp.linspace(1.0, 2.0, n)

A = jnp.diag(diag)
b = jnp.linspace(3.0, 4.0, n)

# A⁻¹ b
jnp.solve(A, b)
```
Doing so is costly in large problems. Storing the matrix gives rise to memory costs of $O(n^2)$, and inverting the matrix costs $O(n^3)$ in the number of data points $n$.

But hold on a second. Notice:

- We only have to store the diagonal entries to determine the matrix $A$. Doing so, would reduce memory costs from $O(n^2)$ to $O(n)$. 
- To invert $A$, we only need to take the reciprocal of the diagonal, reducing inversion costs from $O(n^3)$, to $O(n)$. 

`JaxLinOp` is designed to exploit stucture of this kind. 
```python
import jaxlinop

A = jaxlinop.DiagonalLinearOperator(diag = diag)

# A⁻¹ b
A.solve(b)
```
`JaxLinOp` is designed to automatically reduce cost savings in matrix addition, multiplication, computing log-determinants and more, for other matrix stuctures too!

# Custom Linear Operator (details to come soon)

The flexible design of `JaxLinOp` will allow users to impliment their own custom linear operators.

```python
from jaxlinop import LinearOperator

class MyLinearOperator(LinearOperator):
  
  def __init__(self, ...)
    ...

# There will be a minimal number methods that users need to impliment for their custom operator. 
# For optimal efficiency, we'll make it easy for the user to add optional methods to their operator, 
# if they give better performance than the defaults.
```


# Installation

## Stable version

The latest stable version of `jaxlinop` can be installed via [`pip`](https://pip.pypa.io/en/stable/):

```bash
pip install jaxlinop
```

> **Note**
>
> We recommend you check your installation version:
> ```python
> python -c 'import jaxlinop; print(jaxlinop.__version__)'
> ```



## Development version
> **Warning**
>
> This version is possibly unstable and may contain bugs. 

Clone a copy of the repository to your local machine and run the setup configuration in development mode.
```bash
git clone https://github.com/JaxGaussianProcesses/JaxLinOp.git
cd jaxlinop
python -m setup develop
```

> **Note**
>
> We advise you create virtual environment before installing:
> ```
> conda create -n jaxlinop_ex python=3.10.0
> conda activate jaxlinop_ex
>  ```
>
> and recommend you check your installation passes the supplied unit tests:
>
> ```python
> python -m pytest tests/
> ```
