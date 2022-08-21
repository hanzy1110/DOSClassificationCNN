import os
import pickle
import haiku as hk
import matplotlib.pyplot as plt

from jax import value_and_grad
import jax
import tqdm
import jax.numpy as jnp
import numpy as np

from src.loadDataset import loadDataset, flattenDataset, imbalanceDataset
from src.mtl import TrainingLoopDOS, save, restore, CrossEntropyLoss

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def UpdateWeights(weights,gradients, learning_rate):
    return weights - learning_rate * gradients

class TrainingLoop(TrainingLoopDOS):
    def __init__(self, selectedClases, kj, rj, maxThresh):
        super().__init__(selectedClases, kj, rj, maxThresh)

    def evaluateLoss(self, params_embedder, params_classifier, X_batch, Y_batch):
        embeddings = self.applyEmbedder.apply(params_embedder, self.rng, X_batch)
        preds = self.applyClassifier.apply(params_classifier, self.rng, embeddings)
        return CrossEntropyLoss(preds, Y_batch)

    def trainingLoop(self, applyEmbedder, applyClassifier, 
                     epochs=30, batch_size=256, learning_rate=1/1e4):

        params_classifier, params_embedder = self.instatiateNets(applyEmbedder, applyClassifier, load_prev=False)
        
        with tqdm.tqdm(range(1, epochs+1)) as pbar:

            for i in pbar:
                batches = jnp.arange((self.X_train.shape[0]//batch_size)+1) ### Batch Indices

                losses = [] ## Record loss of each batch
                for batch in batches:
                    if batch != batches[-1]:
                        start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
                    else:
                        start, end = int(batch*batch_size), None

                    X_batch, Y_batch = self.X_train[start:end], self.Y_train[start:end] ## Single batch of data

                    gradFun = value_and_grad(self.evaluateLoss, argnums=(0,1))
                    loss, (params_grads_embedder, params_grads_classifier) = gradFun(params_embedder, params_classifier, X_batch, Y_batch) 

                    params_embedder = jax.tree_map(lambda x,y: UpdateWeights(x, y, learning_rate), 
                                                   params_embedder, params_grads_embedder) ## Update Params
                    params_classifier = jax.tree_map(lambda x,y: UpdateWeights(x, y, learning_rate), 
                                                     params_classifier, params_grads_classifier) ## Update Params


                    losses.append(loss) ## Record Loss
                    
                    if i%10 == 0:
                        self.save_model(params_classifier, params_embedder)

                pbar.set_description("CrossEntropy Loss : {:.3f}".format(jnp.array(losses).mean()))

        fig, ax = plt.subplots(1,1, figsize=(12,8))
        ax.plot(losses)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('CrossEntropy Loss')
        plt.savefig(os.path.join('plots', 'loss.jpg'))
        plt.close(fig)

        self.save_model(params_classifier, params_embedder)
        return params_classifier, params_embedder
