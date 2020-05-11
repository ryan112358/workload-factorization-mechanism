import autograd.numpy as np
from autograd import grad
from autograd.extend import defvjp
from scipy.optimize import bisect
from variance import worst_variance

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

def project_simplex_bounded(r, lb, ub):
    assert lb.sum() <= 1 and ub.sum() >= 1 and np.all(lb <= ub), 'not feasible'
    lambdas = np.append(lb - r, ub - r)
    idx = np.argsort(lambdas)
    lambdas = lambdas[idx]
    active = np.cumsum((idx < r.size)*2 - 1)[:-1]
    diffs = np.diff(lambdas, n=1)
    totals = lb.sum() + np.cumsum(active*diffs)
    i = np.searchsorted(totals, 1.0)
    lam = (1 - totals[i]) / active[i] + lambdas[i+1]
    return np.clip(r + lam, lb, ub)

def project_ldp(Q, z, eps):
    """ Project a strategy onto the feasilbe set

    :param Q: an infeasible strategy
    :param z: minimum allowable value on each row
    :param eps: privacy budget
    :return: a feasible strategy that is as close to Q as possible 
    """
    lb = z
    ub = np.exp(eps)*z
    
    return np.array([project_simplex_bounded(q, lb, ub) for q in Q.T]).T

def initialize(rows, cols, eps):
    """
    Generate a random (feasible) initial strategy of the given shape
    """
    R = np.random.rand(rows, cols)
    low = 1.0 / rows / np.exp(eps)
    high = 1.0 / rows
    z = (low + high) / 2 * np.ones(rows)
    return project_ldp(R, z, eps)

def factorize_tuned(W, eps, Q0, iters=250, verbose=True):
    """
    Optimize the strategy matrix for the given workload and epsilon.
    Use this function if you **do not** have a good guess for the learning rate gamma

    :param W: the workload matrix (p x n)
    :param eps: the privacy budget (scalar)
    :param Q0: the initial strategy (m x n)
    """
    gamma = 1.0
    f = worst_variance(W, Q0, 'opt')
    while gamma > 1e-50:
        try:
            Q = factorize(W, eps, Q0, 50, gamma, verbose)
            break
        except:
            pass
        gamma *= 0.5
    Q1 = Q
    Q2 = factorize(W, eps, Q0, 50, gamma/2, verbose)
    Q3 = factorize(W, eps, Q0, 50, gamma/4, verbose)
    Qs = [Q1, Q2, Q3]
    gammas = [gamma, gamma/2, gamma/4]
    fs = [worst_variance(W, Q, 'opt') for Q in Qs]
    i = np.argmin(fs)
    Q = factorize(W, eps, Qs[i], iters, gammas[i], verbose)
    return Q
    
def factorize(W, eps, Q0, iters=25, gamma=1.0, verbose=True):
    """
    Optimize the strategy matrix for the given workload and epsilon.
    Use this function if you **do** have a good guess for the learning rate gamma

    :param W: the workload matrix (p x n)
    :param eps: the privacy budget (scalar)
    :param Q0: the initial strategy (m x n)
    """
 
    #supports = lambda Q: np.allclose(W.dot(np.linalg.pinv(Q).dot(Q)), W)
    n = Q0.shape[1]
    WtW = W.T @ W
    ww = np.trace(WtW) / n #(W**2).sum(axis=0).mean()
    def average_variance(Q):
        d = np.sum(Q, axis=1)
        QtQ1 = np.linalg.inv(Q.T / d @ Q)
        return QtQ1.flatten() @ WtW.flatten() / n - ww

    gradient = grad(average_variance)
    Q = np.copy(Q0)
    z = Q.min(axis=1)
    c = 1 / n / np.exp(eps)
    best = Q, np.inf
    for t in range(1, iters+1):
        Z1, Z2 = Q == z[:,None], Q == np.exp(eps)*z[:,None]
        f = average_variance(Q)
        if f < best[1]: best = Q, f
        dQ = gradient(Q)
        dz = (dQ*Z1).sum(axis=1) + np.exp(eps)*(dQ*Z2).sum(axis=1)
        #if gamma is None and t == 1: gamma = eps**4 / np.linalg.norm(dz, 1)
        if t % 50 == 0:
            gamma *= 0.5
        if verbose:
            print(t, gamma, f, z.sum(), Q.shape[0])

        z = np.clip(z - c*gamma*dz, 0, 1)
        Q = project_ldp(Q - gamma*dQ, z, eps)
        Q, z = Q[z > 1e-10], z[z > 1e-10]

    #print('best', best[1])
    return best[0]
