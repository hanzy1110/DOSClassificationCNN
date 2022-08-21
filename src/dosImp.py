import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from .loadDataset import (loadDataset, 
                         imbalanceDataset, 
                         TrainingTuple, 
                         flattenDataset, 
                         OverloadedTrainingTuple,
                         OverSampledTrainingTuple)

from typing import Dict, Sequence
from collections import defaultdict
import functools
from operator import iconcat


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

def applyClassifier(x):
    classifier = Classifier()
    return classifier(x)

def computeVYJ(label, 
               labelEmbeddingMap:Dict[int,Sequence[jnp.DeviceArray]], 
               labelTupleMap:Dict[int,Sequence[TrainingTuple]]):
    
    aux = np.array([applyEmbedder(x.image) for x in labelTupleMap[label]])
    labelEmbeddingMap[label] = jnp.array(aux)
    return labelEmbeddingMap

def toImage(t:TrainingTuple):
    return t.image



@functools.partial(jax.jit, static_argnames=["k", "recall_target"])
def l2_ann(query, dataBase, halfDataBaseNorms, k=10, recall_target=0.95):
    dists = halfDataBaseNorms - jax.lax.dot(query, dataBase.transpose())
    return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)

# @jax.jit
@functools.partial(jax.jit, static_argnames=["k"])
def getNeighbors(db, half_db_norms, k, query:jnp.DeviceArray):
    _,neighborsIDX = l2_ann(query=query, 
                            dataBase=db, 
                            halfDataBaseNorms=half_db_norms, 
                            k=k, recall_target=0.95)
    return jnp.array([db[idx] for idx in neighborsIDX])

def getRjVectors(rj, kj):
    key = jax.random.PRNGKey(33)
    arr = jax.random.uniform(key, shape=(rj,kj))
    #L1 normalization
    return arr/arr.sum()
    

class DOSProcedure:
    def __init__(self, selectedClases, kj:Dict[int,int], rj:Dict[int,int], 
                 maxThresh:int, embedder=applyEmbedder)->None:
        self.getDataset(selectedClases, maxThresh=maxThresh)
        self.kj = kj
        self.rj = rj
        self.labelEmbeddingMap = {}
        self.labelWeightVMap = {}
        self.labelNeiMap = defaultdict(lambda: {})
        self.labelOverSampledTupsMap = {}
        self.labelOverloadedTupsMap = {}

        self.embedder = embedder
        init, apply = hk.transform(embedder)
        rng = jax.random.PRNGKey(30)
        params = init(rng, self.X_train[:5])

        # must get a way to refresh params at each iteration
        self.applyEmbedder = lambda x: apply(params, rng, x)
        
    def getDataset(self,selectedClases, maxThresh):

        dataDict = loadDataset()
        self.labelTupleMap = imbalanceDataset(selectedClases, dataDict, maxThresh)
        finalDS = flattenDataset(self.labelTupleMap)

        auxX = np.array([x.image for x in finalDS])
        auxY = np.array([x.label for x in finalDS])
        self.X_train = jnp.array(auxX)
        self.Y_train = jnp.array(auxY)
        
        self.X_test = dataDict['test']['X']
        self.Y_test = dataDict['test']['Y']

    def mainLoop(self,  applyEmbedder, params=None):

        if params:
            # init, applyEmbedder = hk.transform(self.embedder)
            rng = jax.random.PRNGKey(30)
            _ = applyEmbedder.init(rng, self.X_train[:5])
            self.applyEmbedder = lambda x: applyEmbedder.apply(params, rng, x)

        for label, tups in self.labelTupleMap.items():
            # self.labelEmbeddingMap[label] = jnp.array([self.tupleToEmbedding(x) for x in tups])
            images = jnp.array([tup.image for tup in tups])
            # self.labelEmbeddingMap[label] = self.vEmbedder(images)
            self.labelEmbeddingMap[label] = self.applyEmbedder(images)
            self.labelWeightVMap[label] = getRjVectors(self.rj[label], self.kj[label])

        for label, tups in self.labelTupleMap.items():
            db = jnp.array(self.labelEmbeddingMap[label])
            half_db_norms = jax.numpy.linalg.norm(db, axis=1) / 2
            self.vNeighbor = jax.vmap(lambda x: getNeighbors(db=db, half_db_norms=half_db_norms, k=self.kj[label], query=x),
                                      in_axes=(0,))

            # images = jnp.array([tup.image for tup in tups])

            neighbors = self.vNeighbor(db)
            self.labelOverloadedTupsMap[label] = [OverloadedTrainingTuple(**{'image':t.image, 
                                                   'label':t.label, 'neighbors':ns}) for t,ns in zip(tups, neighbors)]

        for label, tups in self.labelOverloadedTupsMap.items():
            wjs = self.labelWeightVMap[label]
            for wj in wjs:
                aux = [self.toOverSampledTuple(tup, wj) for tup in tups]
                # self.labelOverSampledTupsMap[label] = functools.reduce(iconcat, aux, []) 
                self.labelOverSampledTupsMap[label] = aux

        return flattenDataset(self.labelOverSampledTupsMap)
    
    @staticmethod
    def toOverSampledTuple(tup:OverloadedTrainingTuple, wj):
        image, label, neighbors = tup
        params = {'image':image, 'label':label, 
                  'neighbors':neighbors, 'weightVector':wj}
        return OverSampledTrainingTuple(**params)

if __name__ == "__main__":
    qy = jax.numpy.array(np.random.rand(50, 64))
    db = jax.numpy.array(np.random.rand(1024, 64))
    half_db_norms = jax.numpy.linalg.norm(db, axis=1) / 2
    dists, neighbors = l2_ann(qy, db, half_db_norms, k=10)
