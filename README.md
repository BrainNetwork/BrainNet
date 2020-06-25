# Brain Net

##### Software Used

- python v3.7.4
- numpy v1.16.5
- pytorch v1.3.1

## Brain Net options

At the most basic level, there is the `BrainNet` class, taking parameters

- `n` = number of features
- `m` = number of labels/classes
- `num_v` = number of verices in graph
- `p` = probability of an edge in graph
- `rounds` = number of times the graph is applied

There are a few additional options, indicating whether or not Gradient Descent will be performed directly on the weights of the network.

- `full_gd`: perform GD on everything, including graph/output bias and input/graph/output weights.
- `gd_output`: perform GD on output weights
- `gd_input`: perform GD on input weights

These options could be used to set benchmarks.

The `LocalNetBase` class contains the core functionality for the plasticity rule updating scheme. Here, we can pass in an `Options` class that describes which rules are being used and whether or not we should learn them:

```
from LocalNetBase import Options

class Options:
    def __init__(self,
             use_input_rule = False,    # Use an input rule to update input layer weights.
             gd_input_rule = False,     # Learn an input rule, or use fixed one.
                                        # if gd is used, must set 'use_input_rule' to True as well.
             use_output_rule = False,
             gd_output_rule = False,
             use_graph_rule = False,
             gd_graph_rule = False,
             gd_input = False,          # gd directly on input layer weights.
             gd_output = False,         # gd directly on output layer weights.
             additive_rule = True):     # If false, uses multiplicative updates.
        ...
```

We may also specify the updating schemes:

```
from LocalNetBase import UpdateScheme

class UpdateScheme:
    def __init__(self,
            cross_entropy_loss = True,      # By default, uses cross entropy loss.
            mse_loss = False,
            update_misclassified = True,    # By default only update weights when we misclassify.
            update_all_edges = False):      # For output rule, update all edges,
                                            # or only edge to correct and predicted label.
        ...
```

All of these parameters are passed in the `LocalNet` class, a subclass of `LocalNetBase`. Instead of directly instantiating a `LocalNetBase`, we use `LocalNet`. This is cleaner since we can have other classes which also inherit from `LocalNetBase` that have different implementations of the `update_weights` function.

A `LocalNet` can be instantiated simply as follows:

```
from network import LocalNet

local_net = LocalNet(n = dimension, m  = num_labels, num_v = 100, p = .5, cap = 50, rounds = 1,
                            options = options, update_scheme = scheme)
```

## Rule-Based Training

Suppose we have already learned output layer and RNN plasticity rules, `output_rule` and `rnn_rule` repsectively. We can fix these rules by setting the `gd_graph_rule` and `gd_output_rule` to `False`, and assigning them to a `LocalNet` instance as follows. The rules passed in are expected to be tensors.

```
local_net.set_rnn_rule(your_rnn_rule_tensor)
local_net.set_output_rule(torch.tensor([[-1, 1], [1, -1]]))
```

Now, to test this network, we may train using these rules as follows, where `X`, `X_test` are the training and testing data respectively, and `y`, `y_test` are the training/testing labels.

```
from train import train_given_rule

train_acc_learned, test_acc_learned =
    train_given_rule(X, y, local_net,  X_test = X_test, y_test = y_test, verbose = True)
```

The function `train_given_rule` will train the network on the training data using its fixed rules, and test on the testing data. It returns the training accuracies, testing accuracies evaluated every 500 examples (that is, each 500 updates, we evaluate using the current rule). Setting `verbose = False` will only evaluate once after all training is complete.

## Learning Rules with Gradient Descent

To train a `LocalNetBase` instance, say `local_net`, we may use the `train_local_rule` method.

```
from train import train_local_rule

losses, train_accuracies, test_accuracies =
        train_local_rule(
            X,                      # Training Examples
            y,                      # Training Labels
            local_net,              # The BrainNet
            rule_epochs = 1000,     # Number of times to run through the data
            epochs = 1,             # Number of time to run through a single batch
                                    #   for a single training pass
            batch = 100,            # Number of examples in a batch.
            lr = 1e-2,              # Learning Rate
            verbose=False)          # If True, evaluates network on test/train data each epoch
                                    #   otherwise, only evaluates once at the
```

To get the final rules learned as tensors:

```
learned_output_rule = local_net.get_output_rule()
learned_rnn_rule    = local_net.get_rnn_rule()
```

## Data

#### Halfspaces

This generates points labeled by a random linear threshold function.

```
from DataGenerator import random_halfspace_data

# Number of examples to generate for both test and training data
data = 1000

X, y = random_halfspace_data(dim = dimension, n = 2*data)
X_test = X[:data]
y_test = y[:data]
X = X[data:]
y = y[data:]
```

#### ReLu

This generates points labeled by a ReLu feedforward single layer network. To label, takes argmax of the two output nodes.

```
from DataGenerator import layer_relu_data
# Width of the single hidden layer
width = 100

X, y = layer_relu_data(dim = dimension, n = 2*data, k = width)
X_test = X[:data]
Y_test = y[:data]
X = X[data:]
y = y[data:]
```

#### BrainNet

Generates points labeled by a BrainNet

```
from DataGenerator import brainnet_data

X, y = brainnet_data(
            dim = dimension,
            n = 2*data,
            labels = 2,
            num_v = 40,
            p = 0.5,
            cap = 20,
            rounds = 3)
X_test = X[data:]
y_test = y[data:]
X_brain = X[:data]
y_brain = y[:data]
```

#### MNIST

This is the standard MNIST handwritten digit data set. We perform some basic preprocessing of the data first. Each entry in an example is a pixel value from 0 to 255. We divide each entry by 255 to scale everything between 0 and 1.
