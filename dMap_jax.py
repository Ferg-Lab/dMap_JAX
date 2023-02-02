#pip install --upgrade "jax[cpu]"
#pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

from jax import numpy as jnp, jit, vmap

def check_is_symmetric(matrix, tol=1e-8):
    """
    matrix : NxN RMSD matrix
    """
    
    is_symmetric = jnp.all(jnp.abs(matrix-matrix.T) < tol)
       
    return is_symmetric

def identity(x):
    return x

def diffMaps(eps, P, jit_compile=True):
    
    if jit_compile:
        _jit = jit
    else:
        _jit = identity

        
    P = _jit(jnp.exp)(-P**2/(2*eps)) ## A matrix
    
    D = _jit(vmap(jnp.sum, in_axes=(0)))(P)  ### row sums D
    
    inv_sqrtD = D**(-0.5) # inverse square root, other way to calculate #_jit(jnp.sqrt)(D), inv_sqrtD = 1/sqrtD
        
    diag_inv_sqrtD = _jit(jnp.diag)(inv_sqrtD)

    Ms = _jit(jnp.matmul)(_jit(jnp.matmul)(diag_inv_sqrtD, P), diag_inv_sqrtD)
        
    tol = 1e-6
    assert check_is_symmetric(Ms, tol=tol), "Ms should be symmetric!"
        
    lamb, psi = jeigh(Ms)
    
    tol = 1e-8
    assert lamb.min() > tol, "Eigen values less than " + str(tol) + " found."
    
    psi = _jit(jnp.matmul)(diag_inv_sqrtD, psi) # converting evecs of Ms to evecs of M; M and Ms share evals
    
    # return in descending order
    idx_sort = jnp.flip(jnp.argsort(lamb))

    lamb = lamb[idx_sort]
    psi = psi[:,idx_sort]

    
    return lamb, psi