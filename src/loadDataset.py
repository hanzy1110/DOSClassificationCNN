import jax
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass

# from tensorflow import keras
# import tensorflow_datasets as tfds
import keras
from typing import Sequence, Dict, NamedTuple

from jax import Array

from functools import reduce
from operator import iconcat

class TrainingTuple(NamedTuple):
    image:jnp.ndarray
    label:jnp.ndarray

class OverloadedTrainingTuple(NamedTuple):
    image:Array
    label:Array
    neighbors:Array

class OverSampledTrainingTuple(NamedTuple):
    image:Array
    label:Array
    neighbors:Array
    weightVector:Array

def toTraininTuple(image, label):
    return TrainingTuple(image, label)

def getLabelTupleMap(dataDict):
    X_train = dataDict['train']['X']
    Y_train = dataDict['train']['Y']

    trainingTuples = [toTraininTuple(image,label) for image, label in zip(X_train,Y_train)]
    aux = {}
    # TODO add
    for label in range(100):
        aux[label] = [tup for tup in trainingTuples if tup.label==label]

    return aux

def loadDataset()->Dict[str,Dict[str,Array]]:

    # (X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar100.load_data()

    X_train, X_test, Y_train, Y_test = jnp.array(X_train, dtype=jnp.float32),\
                                       jnp.array(X_test, dtype=jnp.float32),\
                                       jnp.array(Y_train, dtype=jnp.float32),\
                                       jnp.array(Y_test, dtype=jnp.float32)

    # X_train, X_test = X_train.reshape(-1,28,28,1), X_test.reshape(-1,28,28,1)
    # X_train, X_test = X_train.reshape(-1,32,32,1), X_test.reshape(-1,32,32,1)

    X_train, X_test = X_train/255.0, X_test/255.0

    return {'train':{'X':X_train, 'Y':Y_train}, 'test':{'X':X_test, 'Y':Y_test}}

def imbalanceDataset(selectedClases:Sequence[int], 
                     dataDict:Dict[str,Dict[str,Array]], maxThresh:int):
    
    labelTupleMap = getLabelTupleMap(dataDict)
    for label in selectedClases:
        tuples = labelTupleMap.pop(label)
        # print(len(tuples))
        labelTupleMap[label] = tuples[:maxThresh]
        print(f'Number Tuples:{len(labelTupleMap[label])}')
    return labelTupleMap

def flattenDataset(labelTupleMap):
    return reduce(iconcat, labelTupleMap.values(), [])

if __name__ == "__main__":
    dataDict = loadDataset()
    imbalancedDataset = imbalanceDataset([1,2,4], dataDict, -1)
    finalDS = flattenDataset(imbalancedDataset)

