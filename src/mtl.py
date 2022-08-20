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

from src.loadDataset import loadDataset, flattenDataset, imbalanceDataset
from src.dosImp import (DOSProcedure, Embedder, Classifier, 
                       ConvNet, applyEmbedder, applyClassifier, OverSampledTrainingTuple)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def save(ckpt_dir: str, state) -> None:
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

def CrossEntropyLoss(preds, actual, classes=10):
    # preds = self.conv_net.apply(weights, self.rng, input_data)
    one_hot_actual = jax.nn.one_hot(actual, num_classes=classes)
    log_preds = jnp.log(preds)
    return - jnp.sum(one_hot_actual * log_preds)

def normParam(preds, neighbors):
    diff = neighbors-preds
    return jnp.sum(jnp.exp(-1*jnp.linalg.norm(diff, axis=0)**2))

def rhoV(embedding, neighbor):
    return jnp.exp(-jnp.linalg.norm(embedding-neighbor)**2)

def embedderLoss(preds, neighbors, actual):
    diff = neighbors-preds
    return jnp.sum(jnp.linalg.norm(diff, axis=0)**2)

def classifierLoss(embedding, actual, neighbors, neighbors_preds, classes=10):
    # Everything is batched!!!!!
    # We need the mappings of the neighbors!
    vRho = jax.vmap(lambda x: rhoV(embedding, x), in_axes=0)
    rhos = vRho(neighbors)
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

    def instatiateNets(self, applyEmbedder, applyClassifier):

        self.rng = jax.random.PRNGKey(42)

        # self.conv_net = hk.transform(CNN)
        self.applyClassifier = hk.transform(applyClassifier)
        self.applyEmbedder = hk.transform(applyEmbedder)

        # paramsCNN = self.conv_net.init(self.rng, self.X_train[:5])
        paramsClassifier = self.applyClassifier.init(self.rng, self.X_train[:5])
        paramsEmbedder = self.applyEmbedder.init(self.rng, self.X_train[:5])

        return paramsClassifier, paramsEmbedder

    def getBatch(self, OSTs:List[OverSampledTrainingTuple], start, end):
        # X_batch, Y_batch = self.X_train[start:end], self.Y_train[start:end] ## Single batch of data
        return OSTs[start:end]

    def getModel(self, epochs=30, batch_size=256, learning_rate=1/1e4):
        if os.path.exists(os.path.join('model', 'arrays.npy')):
            self.params = restore('model')
        else:
            self.params = self.trainingLoop(applyEmbedder, applyClassifier, epochs=epochs, 
                                            batch_size=batch_size, 
                                            learning_rate=learning_rate)

    def evaluateLossEmbedder(self, params, batchData:List[OverSampledTrainingTuple]):

        images = jnp.array([t.image for t in batchData])
        neighbors = jnp.array([t.neighbors for t in batchData])
        labels = jnp.array([t.label for t in batchData])

        embeddings = self.applyEmbedder.apply(params, self.rng, images)
        vLoss = jax.vmap(embedderLoss, in_axes=(0,0,0))

        return jnp.sum(vLoss(embeddings, neighbors, labels))

    def evaluateLossClassifier(self, params_classifier, params_embedder,
                               batchData:List[OverSampledTrainingTuple]):

        images = jnp.array([t.image for t in batchData])
        neighbors = jnp.array([t.neighbors for t in batchData])
        labels = jnp.array([t.label for t in batchData])

        embeddings = self.applyEmbedder.apply(params_embedder, self.rng, images)
        neighbors_preds = self.applyClassifier.apply(params_classifier, self.rng, neighbors)
        preds = self.applyClassifier.apply(params_classifier, self.rng, embeddings)
        vLoss = jax.vmap(classifierLoss, in_axes=(0,0,0,0))

        return jnp.sum(vLoss(embeddings, labels, neighbors,preds, neighbors_preds))

    def trainingLoop(self,applyEmbedder, applyClassifier, 
                     epochs=30, batch_size=256, learning_rate=1/1e4):

        # params = self.conv_net.init(self.rng, self.X_train[:5])
        params_classifier, params_embedder = self.instatiateNets(applyEmbedder, applyClassifier)       

        with tqdm.tqdm(range(1, epochs+1)) as pbar:

            for i in pbar:
                batches = jnp.arange((self.X_train.shape[0]//batch_size)+1) ### Batch Indices
                OSTs = self.dosProc.mainLoop(params=params_embedder)
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
                    loss_classifier, param_grads_classifier = evaluateLossClassifier(params_classifier, params_embedder, ostBatch)
                    loss_embedder, param_grads_embedder = evaluateLossEmbedder(params_embedder, ostBatch)
                    #print(param_grads)
                    params_classifier = jax.tree_map(lambda x,y: UpdateWeights(x, y, learning_rate), 
                                                     params_classifier, param_grads_classifier) ## Update Params

                    params_embedder = jax.tree_map(lambda x,y: UpdateWeights(x, y, learning_rate), 
                                                   params_embedder, param_grads_embedder) ## Update Params


                    losses_classifier.append(loss_classifier) ## Record Loss
                    losses_embedder.append(loss_embedder) ## Record Loss

                if i%10 == 0:
                    pass
                    # save('model', params)

                pbar.set_description("CrossEntropy Loss : {:.3f}".format(jnp.array(losses_embedder).mean()))

#         fig, ax = plt.subplots(1,1, figsize=(12,8))
#         ax.plot(losses)
#         ax.set_xlabel('Iteration')
#         ax.set_ylabel('CrossEntropy Loss')
#         plt.savefig(os.path.join('plots', 'loss.jpg'))
#         plt.close(fig)

        # save('model', params)
        return params_embedder, params_classifier

    def MakePredictions(self, weights, input_data, batch_size=32):
        params = self.conv_net.init(self.rng, self.X_train[:5])
        batches = jnp.arange((input_data.shape[0]//batch_size)+1) ### Batch Indices

        preds = []
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
            else:
                start, end = int(batch*batch_size), None

            X_batch = input_data[start:end]

            preds.append(self.conv_net.apply(weights, self.rng, X_batch))

        return preds

    def predictionAndTest(self):
        train_preds = self.MakePredictions(self.params, self.X_train, 256)
        train_preds = jnp.concatenate(train_preds).squeeze()
        train_preds = train_preds.argmax(axis=1)

        test_preds = self.MakePredictions(self.params, self.X_test, 256)
        test_preds = jnp.concatenate(test_preds).squeeze()
        test_preds = test_preds.argmax(axis=1)


        print("Train Accuracy : {:.3f}".format(accuracy_score(self.Y_train, train_preds)))
        print("Test  Accuracy : {:.3f}".format(accuracy_score(self.Y_test, test_preds)))

        print("Test Classification Report ")
        print(classification_report(self.Y_test, test_preds))

