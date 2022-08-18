import functools
import jax
import numpy as np

@functools.partial(jax.jit, static_argnames=["k", "recall_target"])
def l2_ann(qy, db, half_db_norms, k=10, recall_target=0.95):
    dists = half_db_norms - jax.lax.dot(qy, db.transpose())
    return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)

qy = jax.numpy.array(np.random.rand(50, 64))
db = jax.numpy.array(np.random.rand(1024, 64))
half_db_norms = jax.numpy.linalg.norm(db, axis=1) / 2
dists, neighbors = l2_ann(qy, db, half_db_norms, k=10)
