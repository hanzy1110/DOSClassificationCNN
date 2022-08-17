import os
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

def save_model(params):
    np.savez(os.path.join('model', 'savedModel.npz'), params)

def loadModel():
    return np.loadz(os.path.join('model', 'savedModel.npz'))

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

def CrossEntropyLoss(weights, input_data, actual, classes=10):
    preds = conv_net.apply(weights, rng, input_data)
    one_hot_actual = jax.nn.one_hot(actual, num_classes=classes)
    log_preds = jnp.log(preds)
    return - jnp.sum(one_hot_actual * log_preds)

def UpdateWeights(weights,gradients, learning_rate):
    return weights - learning_rate * gradients

class TrainingLoop:
    def __init__(self, selectedClases, CNN):
        dataDict = loadDataset()
        labelTupleMap = imbalanceDataset(selectedClases, dataDict, -1)
        finalDS = flattenDataset(labelTupleMap)
        self.X_train = jnp.array(np.fromiter(x.getImage() for x in finalDS))
        self.Y_train = jnp.array(np.fromiter(x.getLabel() for x in finalDS))

        self.conv_net = hk.transform(CNN)

    def trainingLoop(self,):
        rng = jax.random.PRNGKey(42) ## Reproducibility ## Initializes model with same weights each time.
        params = self.conv_net.init(rng, self.X_train[:5])
        epochs = 25
        batch_size = 256
        learning_rate = jnp.array(1/1e4)
        
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

                    loss, param_grads = value_and_grad(CrossEntropyLoss)(params, X_batch, Y_batch)
                    #print(param_grads)
                    params = jax.tree_map(lambda x,y: UpdateWeights(x, y, learning_rate), params, param_grads) ## Update Params

                    losses.append(loss) ## Record Loss
                    
                    if i%10 == 0:
                        save_model(params)

                pbar.set_description("CrossEntropy Loss : {:.3f}".format(jnp.array(losses).mean()))

def MakePredictions(weights, input_data, batch_size=32):
    batches = jnp.arange((input_data.shape[0]//batch_size)+1) ### Batch Indices

    preds = []
    for batch in batches:
        if batch != batches[-1]:
            start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
        else:
            start, end = int(batch*batch_size), None

        X_batch = input_data[start:end]

        preds.append(conv_net.apply(weights, rng, X_batch))

    return preds

def predictionAndTest():
    train_preds = MakePredictions(params, X_train, 256)
    train_preds = jnp.concatenate(train_preds).squeeze()
    train_preds = train_preds.argmax(axis=1)

    test_preds = MakePredictions(params, X_test, 256)
    test_preds = jnp.concatenate(test_preds).squeeze()
    test_preds = test_preds.argmax(axis=1)


    print("Train Accuracy : {:.3f}".format(accuracy_score(Y_train, train_preds)))
    print("Test  Accuracy : {:.3f}".format(accuracy_score(Y_test, test_preds)))

    print("Test Classification Report ")
    print(classification_report(Y_test, test_preds))

    plt.plot(losses)
    plt.savefig(os.path.join('plots', 'loss.jpg'))
