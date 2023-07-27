#pip install --upgrade "jax[cpu]"
#pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

from jax import numpy as jnp, jit, vmap
import numpy as np
import pickle
import jax
import os

def jnp_pair_rmsd(ref, target):
    
    '''RMSD of coordinates after alignment using Kabsch algorithm
       https://en.wikipedia.org/wiki/Kabsch_algorithm
    '''
    
    # translation - remove mean
    
    ref = jnp.array(ref)
    target = jnp.array(target)
    
    ref = ref - jnp.mean(ref, axis=0)
    target = target - jnp.mean(target, axis=0)
    
    # covariance
    h = ref.T @ target
    
    # computation of optimal rotation matrix 
    u, s, vh = jnp.linalg.svd(h, full_matrices=False)
    
    #  correct rotation matrix 
    d = jnp.sign(jnp.linalg.det(vh.T @ u.T))
    i = jnp.identity(3)
    i = i.at[(-1,-1)].set(d)
    
    align = vh.T @ i
    align = align @ u.T
    
    target = target @ align # align target to ref
  
    rmsd = jnp.sqrt(jnp.square(ref-target).sum(-1).mean(-1)) # needs update

    return rmsd


def get_pairwise_rmsd_traj(traj, ref_index):
    
    '''RMSD between a trajectory with a given frame.
    '''
    
    # TO do: calculate only upper diagonal i, j pair
    prmsd = jit(vmap(jnp_pair_rmsd, in_axes=(0, None)))(traj, traj[ref_index])
    
    return prmsd


def get_pairwise_rmsd_between_traj(traj1, traj2, ref_index):
    
    '''RMSD between a trajectory with a given frame.
    '''
    
    # TO do: calculate only upper diagonal i, j pair
    prmsd = jit(vmap(jnp_pair_rmsd, in_axes=(0, None)))(traj1, traj2[ref_index])
    
    return prmsd
    

def run_rmsd(traj_jax_array, nref_frames, batch_ref_frame_size=100, output_file_prefix="", device=None, traj2_jax_array=None, overwrite=False):
    
    '''Run RMSD between a trajectory in array format containing n frames with several (defined by batch_ref_frame_size)
    reference frames and save it to a file using pickle
    '''
    last_frame = (nref_frames // batch_ref_frame_size) * batch_ref_frame_size
    cut_traj = traj_jax_array[:last_frame]
    if device:
        set_device =  device
    else:
        set_device =  jax.devices()[0]
    
    count = 0
    
    for i in range(batch_ref_frame_size, nref_frames, batch_ref_frame_size):
    
        ref_indices = jnp.arange(count, i)
        
        if overwrite or not os.path.isfile(output_file_prefix+str(i)+".pkl"):
        # TO do: calculate only upper diagonal i, j pair
            with open(output_file_prefix+str(i)+".pkl", "wb+") as f:

                if traj2_jax_array is None:
                    print(f"running pairwise rmsd between same traj. Batch: {count}")
                    prmsd = jit(vmap(get_pairwise_rmsd_traj, in_axes=(None, 0)), device = set_device)(cut_traj, ref_indices)
                else:
                    print(f"running pairwise rmsd between different traj. Batch: {count}")  
                    prmsd = jit(vmap(get_pairwise_rmsd_between_traj, in_axes=(None, None, 0)), device = set_device)(cut_traj, traj2_jax_array, ref_indices)
                pickle.dump({'rmsd': prmsd}, f)
        else:
            print("Cannot Overwrite " + output_file_prefix+str(i)+".pkl. Try another file name."
    
        count = i
        
        
def load_rmsd(nref_frames, batch_ref_frame_size=100, sym=True, input_file_prefix="", tol=1e-5):
    
    '''Run RMSD between a trajectory in array format containing n frames with several (defined by batch_ref_frame_size)
    reference frames and save it to a file using pickle
    '''
    
    prmsd_jax = []
    for i in range(batch_ref_frame_size, nref_frames, batch_ref_frame_size):
        
        with open(input_file_prefix+str(i)+".pkl", "rb") as f:
            
            data = pickle.load(f)
            prmsd_jax.append(data['rmsd'])
        
    prmsd_jax = np.concatenate(prmsd_jax)
    print(prmsd_jax.shape)
    if sym:
        n_matrix = prmsd_jax.shape[1]
        sym_prmsd_jax = np.zeros((n_matrix, n_matrix))
    
        ## construct full matrix
        upper_diag_prmsd_jax = np.triu_indices_from(sym_prmsd_jax, k=1)
#         print(upper_diag_prmsd_jax.shape)

        sym_prmsd_jax[upper_diag_prmsd_jax] = prmsd_jax.T[upper_diag_prmsd_jax]
        sym_prmsd_jax.T[upper_diag_prmsd_jax] = prmsd_jax.T[upper_diag_prmsd_jax]
        sym_prmsd_jax = jnp.array(sym_prmsd_jax)
    else:
        sym_prmsd_jax = jnp.array(prmsd_jax)

    print("Checking if matrix is symmetric with tol:{}".format(tol))
    print(check_matrix(sym_prmsd_jax, tol=tol))
    
    return sym_prmsd_jax
    
    
def check_is_symmetric(matrix, tol=1e-8):
    """
    matrix : NxN RMSD matrix
    """
    
    is_symmetric = jnp.all(jnp.abs(matrix-matrix.T) < tol)
    
    
    return is_symmetric


def check_matrix(matrix, tol=1e-8):
    """
    matrix : NxN RMSD matrix
    """
    
    diag_min = jnp.diagonal(matrix).min()
    diag_max = jnp.diagonal(matrix).max()
    
    is_symmetric = check_is_symmetric(matrix)
    diff_symmetric_min = jnp.abs(matrix-matrix.T).min()
    diff_symmetric_max = jnp.abs(matrix-matrix.T).max()
    
    
    return ('diag_min: {0}, diag_max: {1}, \n is_symmetric: {2}, diff_symmetric_min:{3}, diff_symmetric_max:{4} '.format(diag_min, diag_max, is_symmetric, diff_symmetric_min, diff_symmetric_max))


def align_rmsd(ref, target):
    
    '''Alignment of coordinates using Kabsch algorithm
       https://en.wikipedia.org/wiki/Kabsch_algorithm
    '''
    
    # translation - remove mean
    
    ref = jnp.array(ref)
    target = jnp.array(target)
    
    ref = ref - jnp.mean(ref, axis=0)
    target = target - jnp.mean(target, axis=0)
    
    # covariance
    h = ref.T @ target
    
    # computation of optimal rotation matrix 
    u, s, vh = jnp.linalg.svd(h, full_matrices=False)
    
    #  correct rotation matrix 
    d = jnp.sign(jnp.linalg.det(vh.T @ u.T))
    i = jnp.identity(3)
    i = i.at[(-1,-1)].set(d)
    
    align = vh.T @ i
    align = align @ u.T
    
    target = target @ align # align target to ref

    return target


    
def run_align_trajs(ref_array, traj_jax_array):

    aligned_traj = jit(vmap(align_rmsd, in_axes=(None, 0)))(ref_array, traj_jax_array)

    return aligned_traj