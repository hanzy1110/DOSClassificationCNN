import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from .loadDataset import loadDataset, imbalanceDataset, TrainingTuple
from typing import Dict, Sequence

import functools

@functools.partial(jax.jit, static_argnames=["k", "recall_target"])
def l2_ann(query, dataBase, halfDataBaseNorms, k=10, recall_target=0.95):
    dists = halfDataBaseNorms - jax.lax.dot(query, dataBase.transpose())
    return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)

class Embedder(hk.Module):
    def __init__(self, classes=10):
        super().__init__(name="Embedder")
        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=(3,3), padding="SAME")
        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=(3,3), padding="SAME")
        self.flatten = hk.Flatten()

    def __call__(self, x_batch):
        x = self.conv1(x_batch)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        return x

class Classifier(hk.Module):
    def __init__(self, classes=10):
        super().__init__(name="Classifier")
        self.linear = hk.Linear(classes)

    def __call__(self, x_batch):
        x = self.linear(x_batch)
        x = jax.nn.softmax(x)
        return x

def ConvNet(x):
    embedder = Embedder()
    classifier = Classifier()
    return classifier(embedder(x))

def applyEmbedder(x):
    embedder = Embedder()
    return embedder(x)

def computeVYJ(label, 
               labelEmbeddingMap:Dict[int,Sequence[jnp.DeviceArray]], 
               labelTupleMap:Dict[int,Sequence[TrainingTuple]]):
    
    aux = np.array([applyEmbedder(x.image) for x in labelTupleMap[label]])
    labelEmbeddingMap[label] = jnp.array(aux)
    return labelEmbeddingMap

# Maybe the below implementation ends up being a bottleneck
@jax.jit
def getNbs(label:int, tup:TrainingTuple, kj:int,
           labelEmbeddingMap:Dict[int,Sequence[jnp.DeviceArray]]):

    query = applyEmbedder(tup.image)
    db = jnp.array(labelEmbeddingMap[label])
    half_db_norms = jax.numpy.linalg.norm(db, axis=1) / 2
    _, neighbors = l2_ann(query, db, half_db_norms, k=kj)
    return neighbors



if __name__ == "__main__":
    qy = jax.numpy.array(np.random.rand(50, 64))
    db = jax.numpy.array(np.random.rand(1024, 64))
    half_db_norms = jax.numpy.linalg.norm(db, axis=1) / 2
    dists, neighbors = l2_ann(qy, db, half_db_norms, k=10)
