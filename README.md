# Jax Linear Operator 
`jax_linear_operator` is a lightweight linear operator library written in [`jax`](https://github.com/google/jax).

# Overview
***<examples / motivation>***

```python
import jax_linear_operator

A = jax_linear_operator.DiagonalLinearOperator(diagonal = jnp.array([1., 2., 3.]))
b = jnp.array([4., 5., 6.])

# A⁻¹ b
A.solve(b)
```

# Custom Linear Operator

***<example / guide>***

```python
from jax_linear_operator import LinearOperator

# This will possibly be a chex dataclass:
class MyLinearOperator(LinearOperator):
  
  def __init__(self, ...)
    ...

# There are a bare mininum number methods that need to be implimented, 
# the user can add optional methods if they are more efficient than defaults given.
```


# Installation

## Stable version

The latest stable version of `jax_linear_operator` can be installed via [`pip`](https://pip.pypa.io/en/stable/):

```bash
pip install jax_linear_operator
```

> **Note**
>
> We recommend you check your installation version:
> ```python
> python -c 'import jax_linear_operator; print(jax_linear_operator.__version__)'
> ```



## Development version
> **Warning**
>
> This version is possibly unstable and may contain bugs. 

Clone a copy of the repository to your local machine and run the setup configuration in development mode.
```bash
git clone https://github.com/Daniel-Dodd/jax_linear_operator.git
cd jax_linear_operator
python setup.py develop
```

> **Note**
>
> We advise you create virtual environment before installing:
> ```
> conda create -n jax_lin_op_ex python=3.10.0
> conda activate jax_lin_op_ex
>  ```
>
> and recommend you check your installation passes the supplied unit tests:
>
> ```python
> python -m pytest tests/
> ```
