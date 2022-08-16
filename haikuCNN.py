import os
import haiku as hk
import matplotlib.pyplot as plt

import jax
from jax import numpy as jnp
import tqdm

from tensorflow import keras
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report




class CNN(hk.Module):
    def __init__(self):
        super().__init__(name="CNN")
        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=(3,3), padding="SAME")
        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=(3,3), padding="SAME")
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(len(classes))

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

def CrossEntropyLoss(weights, input_data, actual):
    preds = conv_net.apply(weights, rng, input_data)
    one_hot_actual = jax.nn.one_hot(actual, num_classes=len(classes))
    log_preds = jnp.log(preds)
    return - jnp.sum(one_hot_actual * log_preds)

(X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

X_train, X_test, Y_train, Y_test = jnp.array(X_train, dtype=jnp.float32),\
                                   jnp.array(X_test, dtype=jnp.float32),\
                                   jnp.array(Y_train, dtype=jnp.float32),\
                                   jnp.array(Y_test, dtype=jnp.float32)

X_train, X_test = X_train.reshape(-1,28,28,1), X_test.reshape(-1,28,28,1)

X_train, X_test = X_train/255.0, X_test/255.0

classes =  jnp.unique(Y_train)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

conv_net = hk.transform(ConvNet)
def UpdateWeights(weights,gradients):
    return weights - learning_rate * gradients

from jax import value_and_grad

rng = jax.random.PRNGKey(42) ## Reproducibility ## Initializes model with same weights each time.

conv_net = hk.transform(ConvNet)
params = conv_net.init(rng, X_train[:5])
epochs = 25
batch_size = 256
learning_rate = jnp.array(1/1e4)

for i in tqdm.tqdm(range(1, epochs+1)):
    batches = jnp.arange((X_train.shape[0]//batch_size)+1) ### Batch Indices

    losses = [] ## Record loss of each batch
    for batch in batches:
        if batch != batches[-1]:
            start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
        else:
            start, end = int(batch*batch_size), None

        X_batch, Y_batch = X_train[start:end], Y_train[start:end] ## Single batch of data

        loss, param_grads = value_and_grad(CrossEntropyLoss)(params, X_batch, Y_batch)
        #print(param_grads)
        params = jax.tree_map(UpdateWeights, params, param_grads) ## Update Params
        losses.append(loss) ## Record Loss
        
        if epoch%10 == 0:
            pass

    print("CrossEntropy Loss : {:.3f}".format(jnp.array(losses).mean()))

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
plt.savefig()
