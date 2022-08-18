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

def save_model(params):
    np.savez(os.path.join('model', 'savedModel.npz'), params)

def loadModel():
    return np.load(os.path.join('model', 'savedModel.npz'), allow_pickle=True)

class CNN(hk.Module):
    def __init__(self, classes=10):
        super().__init__(name="CNN")
        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=(3,3), padding="SAME")
        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=(3,3), padding="SAME")
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(classes)

    def __call__(self, x_batch):
        x = self.conv1(x_batch)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = jax.nn.softmax(x)
        return x

def ConvNet(x):
    cnn = CNN()
    return cnn(x)

def UpdateWeights(weights,gradients, learning_rate):
    return weights - learning_rate * gradients

class TrainingLoop:
    def __init__(self, CNN, selectedClases):
        self.conv_net = hk.transform(CNN)
        self.rng = jax.random.PRNGKey(42) ## Reproducibility ## Initializes model with same weights each time.
        self.getDataset(selectedClases, maxThresh=-1)
        params = self.conv_net.init(self.rng, self.X_train[:5])

    def getDataset(self,selectedClases, maxThresh):
        dataDict = loadDataset()
        labelTupleMap = imbalanceDataset(selectedClases, dataDict, maxThresh=maxThresh)
        finalDS = flattenDataset(labelTupleMap)
        auxX = np.array([x.getImage() for x in finalDS])
        auxY = np.array([x.getLabel() for x in finalDS])
        self.X_train = jnp.array(auxX)
        self.Y_train = jnp.array(auxY)
        self.X_test = dataDict['test']['X']
        self.Y_test = dataDict['test']['Y']

    def getModel(self, epochs=30, batch_size=256, learning_rate=1/1e4):
        if os.path.exists(os.path.join('model', 'arrays.npy')):
            self.params = restore('model')
        else:
            self.params = self.trainingLoop(epochs=epochs, 
                                            batch_size=batch_size, 
                                            learning_rate=learning_rate)

    def CrossEntropyLoss(self, weights, input_data, actual, classes=10):
        preds = self.conv_net.apply(weights, self.rng, input_data)
        one_hot_actual = jax.nn.one_hot(actual, num_classes=classes)
        log_preds = jnp.log(preds)
        return - jnp.sum(one_hot_actual * log_preds)


    def trainingLoop(self,epochs=30, batch_size=256, learning_rate=1/1e4):
        params = self.conv_net.init(self.rng, self.X_train[:5])
        
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

                    loss, param_grads = value_and_grad(self.CrossEntropyLoss)(params, X_batch, Y_batch)
                    #print(param_grads)
                    params = jax.tree_map(lambda x,y: UpdateWeights(x, y, learning_rate), params, param_grads) ## Update Params

                    losses.append(loss) ## Record Loss
                    
                    if i%10 == 0:
                        save('model', params)

                pbar.set_description("CrossEntropy Loss : {:.3f}".format(jnp.array(losses).mean()))

        fig, ax = plt.subplots(1,1, figsize=(12,8))
        ax.plot(losses)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('CrossEntropy Loss')
        plt.savefig(os.path.join('plots', 'loss.jpg'))
        plt.close(fig)

        save('model', params)
        return params

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

