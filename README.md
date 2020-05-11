# workload-factorization-mechanism
Implementation of "A workload-adaptive mechanism for linear queries under local differential privacy"

# Example

Below we optimize the factorization mechanism for the Prefix workload (lower triangular matrix of ones).  We compare with the Hierarchical mechanism.  

First some basic setup:

```
>>> import numpy as np
>>> from mechanisms import hierarchical
>>> from variance import worst_variance
>>> from factorization import factorize_tuned, initialize
>>>
>>> n, eps = 128, 1.0
>>> W = np.tril(np.ones((n,n)))
```

Now get the strategy matrix for the hierarchical mechanism and optimize the factorization mechanism using that as the initialization.

```
>>> Q0 = hierarchical(n, eps) 
>>> worst_variance(W, Q0)
5665.561344360352
>>>
>>> Q = factorize_tuned(W, eps, Q0, verbose=False)
>>> worst_variance(W, Q)
3895.084280703977
```

The factorization mechanism achieves lower worst-case variance than hierarchical in this case by a significant amount.  We can do even better by initializing the optimization with a random strategy matrix (as we recommend in the paper).  

```
>>> np.random.seed(0)
>>> Q0 = initialize(4*n, n, eps) 
>>> Q = factorize_tuned(W, eps, Q0, verbose=False)
>>> worst_variance(W, Q)
1910.8675697859098

```




