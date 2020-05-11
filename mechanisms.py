import numpy as np
from scipy import linalg
import itertools

def hadamard(k, eps):
    """ The hadamard mechanism for general privacy regime
    :param k: the domain size
    :param eps: the privacy budget
    """
    e_eps=np.exp(eps)
    temp=min(e_eps,2*k)
    B= 2**(np.ceil( np.log(temp)/np.log(2) ) -1)
    temp1=(k/B)+1 
    b=2** (np.ceil(np.log(temp1)/np.log(2)) )
    K=int(b*B)
    M=-np.ones((K,K))
    H=linalg.hadamard(int(b))
    ll=[]
    for i in range (int(B)):
        s=i*int(b)
        f=(i+1)*int(b)
        M[s:f,s:f]=H
        ll.append(s)
    M=np.delete(M,ll,1)
    M[np.where(M==1)]=e_eps
    M[np.where(M==-1)]=1
    for i in range(M.shape[1]):
        M[:,i]=M[:,i]/np.sum(M[:,i])

    return M[:,:k]

def randomized_response(k, eps):
    """ the randomized response mechanism
    :param k: the domain size
    :param eps: the privacy budget
    """
    Q = np.ones((k,k)) + (np.exp(eps) - 1) * np.eye(k)
    Q /= np.exp(eps) + k - 1
    return Q

def rappor(k, eps):
    """ the RAPPOR mechanism
    :param k: the domain size
    :param eps: the privacy budget
    """
    p = np.exp(eps/2) / (1.0 + np.exp(eps/2))
    q = 1.0 - p
    binary = np.array(list(itertools.product([0,1], repeat=k)))
    I = np.eye(k)
    Q = np.zeros_like(binary, dtype=np.float64)
    for i in range(k):
        d = np.sum(binary != I[i], axis=1)
        Q[:,i] = p**(k-d) * q**d
    return Q

def hierarchical(k, eps, B=4, oracle=hadamard):
    """ the Hierarchical mechanism for range queries
    :param k: the domain size
    :param eps: the privacy budget
    :param B: the branching factor
    :param oracle: the frequency oracle (e.g., hadamard, randomized_response, etc.)
    """
    result = []
    L = 0
    while B**L < k:
        Ql = oracle(k // B**L, eps)
        T = np.ones(B**L)
        result.append(np.kron(Ql, T))
        L += 1
    return np.vstack(result) / L

# given D and K, return all length D
# binary representation where number of ones <= K
def combo(D, K):
    result = []
    result.append(np.binary_repr(0,D))
    for k in range(1, K + 1):
        comb = list(combinations(list(np.arange(D)), k))
        for i in comb:
            #print(i)
            number = np.zeros(D, dtype=int)
            number[np.array(i)] = 1
            bits = [str(k) for k in number]
            result.append(''.join(bits))

    # Convert to int if necessary
    # result = [int(r, 2) for r in result]

    return result

def convert(N, L):
    bits = bin(N)[2:].zfill(L)
    return [int(k) for k in bits]

def marginalQ(d,k,eps):
    """ Fourier mechanism

    :param d: number of dimensions
    :param k: the order of the marginals of interest
        e.g., 3 for 3 way marginals, d for all marginals, etc.
    :param eps: the privacy budget
    """

    #d=4
    #k=2
    T=combo(d,k) #Set of putput index 
    T_int=[int(r, 2) for r in T] # int value of the output index 
    Input_item=combo(d,d)

    U=2**d
    O=len(T)

    Q=np.zeros((2*O,U))
    H=linalg.hadamard(2**d)
    #eps=1.0
    p= np.exp(eps)/(1+np.exp(eps))
    for i in range(O):
        for j in range (2**d):
            if H[T_int[i],j]==1:
                Q[i,j]=p*(1/O)
                Q[O+i,j]=(1-p)*(1/O)
            else:
                Q[i+O,j]=p*(1/O)
                Q[i,j]=(1-p)*(1/O)
    return Q

def run(Q, u, prng=np.random):
    """ Run the mechanism 

    :param Q: the strategy matrix
    :param u: the vector of user types
    :param prng: random number generator
    """
    m, n = Q.shape
    o = np.array([prng.choice(m, p=Q[:,v]) for v in u])
    x = np.bincount(u, minlength=n)
    y = np.bincount(o, minlength=m)
    info = { 'x' : x, 'y' : y }
    return o, info
