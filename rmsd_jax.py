#pip install --upgrade "jax[cpu]"
#pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

from jax import numpy as jnp, jit, vmap


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
    
    prmsd = jit(vmap(jnp_pair_rmsd, in_axes=(0, None)))(traj, traj[ref_index])
    
    return prmsd
    

def run_rmsd(traj_jax_array, batch_ref_frame_size=100, output_file_prefix="pair_rmsd_data/pair_rmsd_break_legs_no_break_"):
    count = 0
    for i in range(100, traj_jax_array.shape[0]+1, 100):
    
        ref_indices = jnp.arange(count, i)

        with open(output_file_prefix+str(i)+".pkl", "wb") as f:
            prmsd = jit(vmap(get_pairwise_rmsd_traj, in_axes=(None, 0)))(traj_jax_array, ref_indices)
            pickle.dump({'rmsd': prmsd}, f)
    
        count = i