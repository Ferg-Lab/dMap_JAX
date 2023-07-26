#pip install --upgrade "jax[cpu]"
#pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

import jax
from jax import numpy as jnp, jit, vmap
from jax.numpy.linalg import eigh as jeigh
from jax.scipy.linalg import eigh as jseigh
from scipy.linalg import eigh as eigh
import numpy as np

def check_is_symmetric(matrix, tol=1e-8):
    """
    matrix : NxN RMSD matrix
    """
    
    is_symmetric = jnp.all(jnp.abs(matrix-matrix.T) < tol)
       
    return is_symmetric

def identity(x, device=None):
    return x

def diffMaps(eps, P, alpha=None, jit_compile=True, check_pos_definite=True, tol_sym=1e-6, tol=1e-9, device=None):
        
    if device:
        set_device =  device
    else:
        set_device =  jax.devices()[0]
         
    
    if jit_compile:
        _jit = jit
    else:
        _jit = identity


    if alpha != None:
        print("Applying adaptive density diffusion map using alpha:{}".format(alpha))
        P = _jit(jnp.exp, device=device)((-P**(2*alpha))/(2*eps)) ## A matrix
    else:
        print("Applying normal diffusion map as alpha is :{}".format(alpha))
        P = _jit(jnp.exp, device=device)((-P**2)/(2*eps)) ## A matrix
    
    D = _jit(vmap(jnp.sum, in_axes=(0)), device=device)(P)  ### row sums D
    
    inv_sqrtD = D**(-0.5) # inverse square root, other way to calculate #_jit(jnp.sqrt)(D), inv_sqrtD = 1/sqrtD
        
    diag_inv_sqrtD = _jit(jnp.diag, device=device)(inv_sqrtD)

    Ms = _jit(jnp.matmul, device=device)(_jit(jnp.matmul, device=device)(diag_inv_sqrtD, P), diag_inv_sqrtD)
        
    assert check_is_symmetric(Ms, tol=tol_sym), "Ms should be symmetric!"
        
    lamb, psi = jseigh(Ms)
    
    
    if check_pos_definite:
        assert lamb.min() > tol, "Eigen values less than " + str(tol) + " found. Min eigval is " + str(lamb.min())
    
    psi = _jit(jnp.matmul, device=device)(diag_inv_sqrtD, psi) # converting evecs of Ms to evecs of M; M and Ms share evals
    

    # return in descending order
    idx_sort = jnp.flip(jnp.argsort(lamb))

    lamb = lamb[idx_sort]
    psi = psi[:,idx_sort]

    
    return lamb, psi


def nystrom_jax(eps, Pnew, lamb, psi, alpha=None, jit_compile=True, device=None):
    #delayVecs,save_bin,lamb,psi,eps,delayVecs_ss):
    
    if device:
        set_device =  device
    else:
        set_device =  jax.devices()[0]
    
    if jit_compile:
        _jit = jit
    else:
        _jit = identity
             
    if alpha != None:
        print("Applying adaptive density diffusion map using alpha:{}".format(alpha))
        Pnew = _jit(jnp.exp, device=device)((-Pnew**(2*alpha))/(2*eps)) ## A matrix
    else:
        print("Applying normal diffusion map as alpha is :{}".format(alpha))
        Pnew = _jit(jnp.exp, device=device)((-Pnew**2)/(2*eps)) ## A matrix
    
    D = _jit(vmap(jnp.sum, in_axes=(0)), device=device)(Pnew)  ### row sums D
    
    inv_sqrtD = D**(-0.5) # inverse square root, other way to calculate #_jit(jnp.sqrt)(D), inv_sqrtD = 1/sqrtD
        
    diag_inv_sqrtD = _jit(jnp.diag, device=device)(inv_sqrtD)

    Mnew = _jit(jnp.matmul, device=device)(_jit(jnp.matmul, device=device)(diag_inv_sqrtD, Pnew), diag_inv_sqrtD)
    
    print(jnp.amax(Mnew, axis=1))
        
    #diag_inv_sqrtD = _jit(jnp.diag, device = set_device)(inv_sqrtD)
    
    print('Commencing Nystrom extension...')
       
    psi_new = _jit(jnp.matmul, device=device)(Mnew, psi)
    
    psi_new = jnp.divide(psi_new, lamb)
    
    print('Completed.')
    
    return jnp.array(psi_new)

 
## Codes from Max - pivot-dMaps ("http://dx.doi.org/10.1021/acs.macromol.7b01684")
def pivot_extract(n_points, dist, r_cut, alpha, pivot_file):
    
    #SYMMETRIZED VERSION
    print("Starting pivot...")

    N = n_points
    print(f"Finding the pivot for {N} data points")
    Pivot = np.zeros(N)

    n=-1
    #determine [position of all pivots]
    index_pivot = []
    start = time.time()
    for i in range(N):
        if Pivot[i]==0:
            Pivot[i]=1
            n+=1
            index_pivot.append(i)
            #cutoff adjustment following the advice of the original paper
            temp = np.power(dist[i,:],alpha) #0.9*np.power(dist[i,:],alpha)
            Pivot = np.where(np.logical_and(temp <= r_cut,Pivot==0),2,Pivot)
        if (i%100==0):
            print(100*i/N," % Done")

    print("Pivot finished in {} seconds".format(time.time()-start))
    
    #print("Pivot Done!")
    with open(pivot_file+".pkl", "wb") as f:
        pickle.dump({'pivot': Pivot, 'index_pivot': index_pivot, 'n': n}, f)
            

    return index_pivot, n, Pivot


## Codes from Max - Out of sample projection using Nystrom extension
def nystrom(delayVecs,save_bin,lamb,psi,eps,delayVecs_ss):
    
    start = time.time()
    
    #create saving bins for nystrom extension every save_bin frames
    panels_indx = np.arange(0,len(delayVecs),save_bin)
    #append the last frame if not already in lise
    panels_indx = list(panels_indx)
    if len(delayVecs)!=panels_indx[-1]:
        panels_indx.append(len(delayVecs))
    print(panels_indx)
    
    #check to see if any have already been run
    k = 0
    for q in range(len(panels_indx)):
        if os.path.isfile('data_%d.npz' %(q)):
            k=q
        else:
            break
    
    #only run nystrom on problems that we havenet run on before
    for j in range(k,len(panels_indx)-1): # subsegments 
    #for j in range(3*traj.n_frames//10-1,traj.n_frames,traj.n_frames//10):
    
        print('Commencing Nystrom extension %d...' %(j))
        z = []
        #for i in range(j,j+traj.n_frames//10):
     #   for i in range(panels_indx[j],panels_indx[j+1]):
         
        for i in range(delayVecs.shape[0]): # number of total points (but can be only new points)
            Euc_i = np.minimum( euclidean_distances(delayVecs_ss,delayVecs[i].reshape(1,-1)), euclidean_distances(delayVecs_ss_flip,delayVecs[i].reshape(1,-1)) ) # for symmetries (minimum distances)

            Euc_i = np.exp(-Euc_i**2/(2*eps))

            Euc_i /= np.sum(Euc_i)

            psi_Nystrom = np.divide( np.matmul(Euc_i.T,psi), lamb)

            z.append(psi_Nystrom)

            
            #report every 10000 frames
            if np.mod(i+1,10000) == 0:
                print('\tNystromed %d of %d frames...' % (i+1,len(delayVecs)))
        #save every d frames
        #np.savez('data_%d.npz' %(j), z=z, j=j)

    print('DONE!')
    print('')

    end = time.time()
    print("Elapsed time %.2f (s)" % (end - start))
    return z


def pivot_dnystrom(dist,n,index_pivot):
    #construct dnystrom from pivots 
    P = np.zeros((n+1,n+1),dtype=float)

    for i in range(n+1):
        for j in range(n+1):
            P[i,j] = dist[index_pivot[i],index_pivot[j]]

    # Now we have a reduced for distance matrix composed only of pivot points
    # we will now 1) run dmap on this
    #2) Run a nystrom extension

    outfile='P.npz'
    np.savez(outfile, P=P)
    return P

## Code from Max 
def diffMaps_Numpy(eps,P, alpha):

    P = np.exp((-P**(2*alpha))/(2*eps))
    D = np.sum(P,axis=1)
    P = np.matmul(np.matmul(np.diag(D**(-0.5)),P),np.diag(D**(-0.5))) # constructing Ms as symmetric matrix for diagonalization

    #nEvals=20
    lamb, psi = eigh(P) #,eigvals=(P.shape[0]-nEvals,P.shape[0]-1) # scipy eigh to specify only computation of leading evals
    psi = np.matmul(np.diag(D**(-0.5)),psi) # converting evecs of Ms to evecs of M; M and Ms share evals

    idx_sort = np.flip(np.argsort(lamb))

    lamb = lamb[idx_sort]
    psi = psi[:,idx_sort]
    
    return lamb, psi
