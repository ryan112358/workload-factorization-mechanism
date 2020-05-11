import autograd.numpy as np
from autograd.extend import defvjp
import itertools
import warnings

def pinv_vjp(ans, A):
    A1 = np.linalg.pinv(A)
    In = np.eye(A.shape[1])
    Im = np.eye(A.shape[0])
    def foo(g):
        term1 = -np.dot(A1, np.dot(g.T, A1))
        term2 = np.dot(np.dot(A1, A1.T), np.dot(g, Im - np.dot(A, A1)))
        term3 = np.dot(In - np.dot(A1, A), np.dot(g, np.dot(A1.T, A1)))
        return (term1 + term2 + term3).T
    return foo

defvjp(np.linalg.pinv, pinv_vjp) 
def magic_vector(W, Q, V='pinv'):
    # return the vector z, such that x^T * z is the total variance on the workload
    if V is 'pinv':
        Q1 = np.linalg.pinv(Q, rcond=1e-8)
        V = W @ Q1
    elif V is 'opt':
        d = Q.sum(axis=1)
        QtQ1 = np.linalg.pinv( Q.T / d @ Q, rcond=1e-8, hermitian=True )
        V = W @ QtQ1 @ Q.T / d
    if not np.allclose(W, V @ Q):
        err = np.linalg.norm(W - V @ Q)
        warnings.warn('not a valid workload factorization, %.2f' % err)
    Z = np.dot(V**2, Q) - W**2
    return Z.sum(axis=0)

def total_variance(W, Q, x, V='pinv'):
    """ return the total variance on the workload queries, given the 
        strategy matrix Q and the data-vector x

    :param W: the m x k workload matrix
    :param Q: the K x k mechanism matrix
    :param x: the k-length data-vector
    :return: the total variance on the workload
    """
    z = magic_vector(W, Q, V)
    return np.dot(x, z) / x.sum()

def average_variance(W, Q, V='pinv'):
    """ return the expected variance on the workload queries, 
        averaged over all data-vectors, given the strategy matrix Q
    :param W: the m x k workload matrix
    :param Q: the K x k mechanism matrix
    :return: the average-case variance on the workload
    """
    z = magic_vector(W, Q, V)
    return np.mean(z)

def worst_variance(W, Q, V='pinv'):
    """ return the worst-case variance on the workload queries, 
        over all data-vectors, given the strategy matrix Q
    :param W: the m x k workload matrix
    :param Q: the K x k mechanism matrix
    :return: the worst-case variance on the workload
    """
    z = magic_vector(W, Q, V)
    return z.max()

def lower_bound(W, eps):
    n = W.shape[1]
    w = np.diag(W.T @ W)
    p = np.sqrt(w)
    p *= np.exp(eps) / p.sum()
    return ( np.sum(w / p) - np.sum(w) ) / n
