import os
import pickle
import haiku as hk
import matplotlib.pyplot as plt

from jax import value_and_grad
import jax
import tqdm
import jax.numpy as jnp
import numpy as np
from typing import List, Dict

from functools import partial

from src.loadDataset import loadDataset, flattenDataset, imbalanceDataset
from src.dosImp import (DOSProcedure, Embedder, Classifier, 
                       ConvNet, applyEmbedder, applyClassifier, OverSampledTrainingTuple)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def save(ckpt_dir: str, state) -> None:
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)

def restore(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_unflatten(treedef, flat_state)

@jax.jit
def CrossEntropyLoss(preds, actual, classes=10):
    # preds = self.conv_net.apply(weights, self.rng, input_data)
    one_hot_actual = jax.nn.one_hot(actual, num_classes=classes)
    log_preds = jnp.log(preds)
    return - jnp.sum(one_hot_actual * log_preds)

@jax.jit
def normParam(preds, neighbors):
    diff = neighbors-preds
    return jnp.sum(jnp.exp(-1*jnp.linalg.norm(diff, axis=0)**2))

@jax.jit
def rhoV(embedding, neighbor, weight):
    return jnp.exp(-weight * jnp.linalg.norm(embedding-neighbor)**2)

@jax.jit
def embedderLoss(preds, neighbors, actual, weight):
    diff = neighbors-preds
    # return jnp.sum(jnp.linalg.norm(diff, axis=0)**2)
    return jnp.dot(jnp.linalg.norm(diff, axis=0)**2, weight)

@jax.jit
def classifierLoss(embedding, actual, neighbors, neighbors_preds, weights, classes=10):
    # Everything is batched!!!!!
    # We need the mappings of the neighbors!
    vRho = jax.vmap(lambda x, y: rhoV(embedding, x, y), in_axes=(0,0))
    rhos = vRho(neighbors, weights)
    Z = normParam(embedding, neighbors)
    vCrossEnt = jax.vmap(lambda x: CrossEntropyLoss(x, actual), in_axes=0)
    # H(g(v)) for each neighbor
    crossEntropies = vCrossEnt(neighbors_preds)
    return jnp.dot(rhos/Z, crossEntropies)

def UpdateWeights(weights,gradients, learning_rate):
    return weights - learning_rate * gradients

class TrainingLoopDOS:
    def __init__(self, selectedClases, kj, rj, maxThresh)->None:

        self.dosProc = DOSProcedure(selectedClases, kj, rj, maxThresh)
        self.X_train = self.dosProc.X_train
        self.Y_train = self.dosProc.Y_train
        self.X_test = self.dosProc.X_test
        self.Y_test = self.dosProc.Y_test
        self.wasInitiated = False

    def instatiateNets(self, applyEmbedder, applyClassifier, load_prev:bool, ckpt_dir='model'):

        self.rng = jax.random.PRNGKey(42)

        # self.conv_net = hk.transform(CNN)
        self.applyClassifier = hk.transform(applyClassifier)
        self.applyEmbedder = hk.transform(applyEmbedder)

        # paramsCNN = self.conv_net.init(self.rng, self.X_train[:5])
        paramsEmbedder = self.applyEmbedder.init(self.rng, self.X_train[:5])
        # paramsClassifier = self.applyClassifier.init(self.rng, self.X_train[:5])
        test = self.applyEmbedder.apply(paramsEmbedder, self.rng, self.X_train[:5])
        paramsClassifier = self.applyClassifier.init(self.rng, test)

        self.wasInitiated = True
        if load_prev:
            paramsClassifier, paramsEmbedder = self.restore_model(ckpt_dir)

        return paramsClassifier, paramsEmbedder

    def getBatch(self, OSTs:List[OverSampledTrainingTuple], start, end):
        # X_batch, Y_batch = self.X_train[start:end], self.Y_train[start:end] ## Single batch of data
        return OSTs[start:end]

    def getModel(self, applyEmbedder, applyClassifier, 
                 epochs=30, batch_size=256, learning_rate=1/1e4, ckpt_dir='model'):

        if os.path.exists(os.path.join('model',  'classifier', 'arrays.npy')):
            self.params_classifier, self.params_embedder = self.restore_model(ckpt_dir)
        else:
            self.params_classifier, self.params_embedder = self.trainingLoop(applyEmbedder, applyClassifier,
                                                                             epochs=epochs, 
                                                                            batch_size=batch_size, 
                                                                            learning_rate=learning_rate)

    def save_model(self, params_classifier, params_embedder, ckpt_dir='model'):
        save(os.path.join(ckpt_dir, 'classifier'), params_classifier)
        save(os.path.join(ckpt_dir, 'embedder'), params_embedder)

    def restore_model(self, ckpt_dir='model'):
        params_classifier = restore(os.path.join(ckpt_dir, 'classifier'))
        params_embedder = restore(os.path.join(ckpt_dir, 'embedder'))
        return params_classifier, params_embedder

    # @partial(jax.jit, static_argnums=0)
    def evaluateLossEmbedder(self, params, batchData:List[OverSampledTrainingTuple]):

        images = jnp.array([t.image for t in batchData])
        neighbors = jnp.array([t.neighbors for t in batchData])
        labels = jnp.array([t.label for t in batchData])
        weights = jnp.array([t.weightVector for t in batchData])

        embeddings = self.applyEmbedder.apply(params, self.rng, images)
        vLoss = jax.vmap(embedderLoss, in_axes=(0,0,0,0))

        # Take the average over the batch
        return jnp.sum(vLoss(embeddings, neighbors, labels, weights))/images.shape[0]

    # @partial(jax.jit, static_argnums=0)
    def evaluateLossClassifier(self, params_classifier, params_embedder,
                               batchData:List[OverSampledTrainingTuple]):

        images = jnp.array([t.image for t in batchData])
        neighbors = jnp.array([t.neighbors for t in batchData])
        labels = jnp.array([t.label for t in batchData])
        weights = jnp.array([t.weightVector for t in batchData])

        embeddings = self.applyEmbedder.apply(params_embedder, self.rng, images)

        vClass = jax.vmap(lambda x: self.applyClassifier.apply(params_classifier, self.rng, x), in_axes=1)

        # neighbors_preds = self.applyClassifier.apply(params_classifier, self.rng, neighbors)
        neighbors_preds = vClass(neighbors)
        # preds = self.applyClassifier.apply(params_classifier, self.rng, embeddings)
        vLoss = jax.vmap(classifierLoss, in_axes=(0,0,0,1,0))

        return jnp.sum(vLoss(embeddings, labels, neighbors, neighbors_preds, weights))

    def trainingLoop(self,applyEmbedder, applyClassifier, 
                     epochs=30, batch_size=256, learning_rate=1/1e4, ckpt_dir='model'):

        # params = self.conv_net.init(self.rng, self.X_train[:5])
        print("Starting Training...")
        if not self.wasInitiated:
            params_classifier, params_embedder = self.instatiateNets(applyEmbedder, 
                                                                     applyClassifier, 
                                                                     load_prev=True)       

        with tqdm.tqdm(range(1, epochs+1)) as pbar:

            for i in pbar:
                batches = jnp.arange((self.X_train.shape[0]//batch_size)+1) ### Batch Indices
                OSTs = self.dosProc.mainLoop(params=params_embedder, applyEmbedder=self.applyEmbedder)
                losses_embedder = [] ## Record loss of each batch
                losses_classifier = [] ## Record loss of each batch

                for batch in batches:
                    if batch != batches[-1]:
                        start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
                    else:
                        start, end = int(batch*batch_size), None

                    ostBatch = self.getBatch(OSTs, start, end)

                    # Get the predictions outside both models!
                    # First argument must be the weights to take the gradients with respect to!
                    evaluateLossEmbedder = value_and_grad(self.evaluateLossEmbedder)
                    evaluateLossClassifier = value_and_grad(self.evaluateLossClassifier)

                    loss_embedder, param_grads_embedder = evaluateLossEmbedder(params_embedder, ostBatch)
                    loss_classifier, param_grads_classifier = evaluateLossClassifier(params_classifier, params_embedder, ostBatch)

                    params_classifier = jax.tree_map(lambda x,y: UpdateWeights(x, y, learning_rate), 
                                                     params_classifier, param_grads_classifier) ## Update Params

                    params_embedder = jax.tree_map(lambda x,y: UpdateWeights(x, y, learning_rate), 
                                                   params_embedder, param_grads_embedder) ## Update Params


                    losses_classifier.append(loss_classifier) ## Record Loss
                    losses_embedder.append(loss_embedder) ## Record Loss

                if i%10 == 0:
                    self.save_model(params_classifier, params_embedder, ckpt_dir=ckpt_dir)

                pbar.set_description("CrossEntropy Loss : {:.3f}".format(jnp.array(losses_embedder).mean()))

        fig, ax = plt.subplots(1,1, figsize=(12,8))
        ax.plot(losses_embedder, label="Embedder Loss")
        ax.plot(losses_classifier, label="Classifier Loss")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        plt.savefig(os.path.join('plots', 'loss.jpg'))
        plt.close(fig)

        self.save_model(params_classifier, params_embedder, ckpt_dir=ckpt_dir)
        return params_embedder, params_classifier

    def MakePredictions(self, params_classifier, params_embedder, input_data, batch_size=32):
        batches = jnp.arange((input_data.shape[0]//batch_size)+1) ### Batch Indices

        preds = []
        for batch in batches[:-1]:
            if batch != batches[-1]:
                start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
            # else:
            #     start, end = int(batch*batch_size), None

            X_batch = input_data[start:end]

            embeddings = self.applyEmbedder.apply(params_embedder, self.rng, X_batch)
            preds.append(self.applyClassifier.apply(params_classifier, self.rng, embeddings))

        return preds

    def predictionAndTest(self, ckpt_dir='model'):
        if not self.wasInitiated:
            _, _ = self.instatiateNets(applyEmbedder, applyClassifier, load_prev=False)       
        params_classifier, params_embedder = self.restore_model(ckpt_dir=ckpt_dir)

        train_preds = self.MakePredictions(params_classifier, params_embedder, self.X_train, 256)
        train_preds = jnp.concatenate(train_preds).squeeze()
        train_preds = train_preds.argmax(axis=1)

        test_preds = self.MakePredictions(params_classifier, params_embedder, self.X_test, 256)
        test_preds = jnp.concatenate(test_preds).squeeze()
        test_preds = test_preds.argmax(axis=1)

        print("Train Accuracy : {:.3f}".format(accuracy_score(self.Y_train, train_preds)))
        print("Test  Accuracy : {:.3f}".format(accuracy_score(self.Y_test[:test_preds.shape[0]], test_preds)))

        print("Test Classification Report ")
        print(classification_report(self.Y_test[:test_preds.shape[0]], test_preds))
