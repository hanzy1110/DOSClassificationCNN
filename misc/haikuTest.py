from absl import app
import haiku as hk

import jax
import optax
import numpy as np

from typing import Iterator, NamedTuple

from jax.example_libraries import stax, optimizers
from jax import numpy as jnp
from jax import value_and_grad

from tensorflow import keras
from sklearn.model_selection import train_test_split

class TrainingState(NamedTuple):
  params: hk.Params
  avg_params: hk.Params
  opt_state: optax.OptState

class Batch(NamedTuple):
  image: np.ndarray  # [B, H, W, 1]
  label: np.ndarray  # [B]


# class innerDense(hk.Module):
#     def __init__(self, classes):
#         self.classes = classes
#         super(innerDense,self).__init__()

#     def __call__(self, x):

#         mlp = hk.Sequential([
#             hk.Linear(300), jax.nn.relu,
#             hk.Linear(100), jax.nn.relu,
#             hk.Linear(self.classes),
#         ])

#         return mlp(x)
        
# class CNN(hk.Module):
#     def __init__(self, classes):
#         self.classes = classes
#         super(CNN,self).__init__()

#     def __call__(self, x):
        
#         cnn = hk.Sequential(
#             [hk.Conv2D(32,(3,3), padding="SAME"),
#             jax.nn.relu,
#             hk.Conv2D(16, (3,3), padding="SAME"),
#             jax.nn.relu,
#             hk.Flatten,
#             innerDense(len(self.classes)),
#             jax.nn.softmax],
#                     )
#         return cnn(x)

class CNN(hk.Module):
    def __init__(self, classes):
        
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

def MakePredictions(weights, input_data, batch_size=32):
    batches = jnp.arange((input_data.shape[0]//batch_size)+1) ### Batch Indices

    preds = []
    for batch in batches:
        if batch != batches[-1]:
            start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
        else:
            start, end = int(batch*batch_size), None

        X_batch = input_data[start:end]

        if X_batch.shape[0] != 0:
            preds.append(conv_net.apply(weights, X_batch))

    return preds

# weights = weights[1] ## Weights are actually stored in second element of two value tuple

# seed = jax.random.PRNGKey(123)
# learning_rate = jnp.array(1/1e4)
# epochs = 25
# batch_size=256

# weights = conv_init(rng, (batch_size,28,28,1))
# weights = weights[1]


def main(_):
    optimiser = optax.adam(1e-3)

    def loss(params:hk.Params, batch:Batch, actual):
        # batch_size, *_ = batch.image.shape
        logits = conv_net.apply(params, batch.image)
        one_hot_actual = jax.nn.one_hot(actual, num_classes=10)
        return - jnp.sum(one_hot_actual * logits)


    @jax.jit
    def evaluate(params: hk.Params, batch: Batch) -> jnp.ndarray:
        """Evaluation metric (classification accuracy)."""
        logits = conv_net.apply(params, batch.image)
        predictions = jnp.argmax(logits, axis=-1)
        return jnp.mean(predictions == batch.label)

    @jax.jit
    def update(state: TrainingState, batch: Batch) -> TrainingState:
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(loss)(state.params, batch)
        updates, opt_state = optimiser.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        # Compute avg_params, the exponential moving average of the "live" params.
        # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
        avg_params = optax.incremental_update(
            params, state.avg_params, step_size=0.001)
        return TrainingState(params, avg_params, opt_state)

    (X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

    X_train, X_test, Y_train, Y_test = jnp.array(X_train, dtype=jnp.float32),\
                                       jnp.array(X_test, dtype=jnp.float32),\
                                       jnp.array(Y_train, dtype=jnp.float32),\
                                       jnp.array(Y_test, dtype=jnp.float32)

    X_train, X_test = X_train.reshape(-1,28,28,1), X_test.reshape(-1,28,28,1)

    X_train, X_test = X_train/255.0, X_test/255.0

    classes =  jnp.unique(Y_train)

    def ConvNet(x):
        cnn = CNN(len(classes))
        return cnn(x)

    rng = jax.random.PRNGKey(123)
    conv_net = hk.without_apply_rng(hk.transform(ConvNet))
    # initBatch = Batch(**{'image':X_train[0,...], 'label':Y_train[0,...]})
    # initial_params = fCNN.init(rng, initBatch.image)
    initial_params = conv_net.init(rng, X_train[:5])

    initial_opt_state = optimiser.init(initial_params)
    state = TrainingState(initial_params, initial_params, initial_opt_state)

    # Trining & evaluation loop.
    for step in range(3001):
        # Periodically evaluate classification accuracy on train & test sets.
        # Note that each evaluation is only on a (large) batch.
        for xtrain,ytrain,xtest,ytest in zip(X_train, X_test, Y_train, Y_test):

            batch = Batch(**{'image':xtrain, 'label':ytrain,})
            if step % 100 == 0:
                accuracy = np.array(evaluate(state.avg_params, batch)).item()
                print({"step": step, "split": "None", "accuracy": f"{accuracy:.3f}"})

            # Do SGD on a batch of training examples.
            state = update(state,batch)

if __name__ == "__main__":
    app.run(main)
