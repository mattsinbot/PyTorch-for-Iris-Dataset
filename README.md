# PyTorch-for-Iris-Dataset
A simple stand alone project of deep learning using PyTorch to solve classification problem. We build a multilayer deep neural net first and train it using Iris data set obtained from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data . 

## PyTorch installation
A good introduction to PyTorch hands-on examples can be found https://github.com/jcjohnson/pytorch-examples. In this repository, we intent to show one complete example with PyTorch with high prediction accuracy. To start with, install PyTorch using Anaconda. The current version of PyTorch is ```1.0.0```. In order to install PyTorch use the following command,

```$ conda install -c pytorch pytorch```

To check if installation was successful, in a terminal type in the following,

```
$ python
>>> import torc
>>> print("PyTorch version: {}".format(torch.__version__))
```

For complete implementation, please look at the jupyter notebook file, named as **classify_Iris_data.ipynb**.

## Data set loading
The main data set is named as **iris_dataset.txt** which we do not use in the code. Instead we have prepared two different data sets, named **iris_train_dataset.txt** and **iris_test_dataset.txt** for training and testing the model. The training dataset has 120 examples whereas testdataset has 30 test examples. The data set has dimensionality 4, namely sepal_length, sepal_width, petal_length, petal_width. The target labels are one of the 3 classes of Iris, namely **iris setosa**, **iris versicolor**, **iris virginica** respectively. In order to have numeric value of these classes, we use **One-Hot** encoding so that
```
iris setosa     = [1,0,0]
iris versicolor = [0,1,0] 
iris verginica  = [0,0,1]
```
**Note**: After loading the data in Python environment, we need to convert them to PyTorch tensors using the command ```torch.Tensor(data)```.

## Data Processing
In order to have comparable effects of all features on to the model, we normalize each feature by

```
normalized_feature = (actual_feature - mean)/std
```
Remember, if you use normalized features for training the network, you have to use normalized features for testing the model as well. Otherwise you will be comparing apple with orange, predicting wrong results.
**Note**: We should always use normalized features to train the network not just for the above reason, there is a practical angle to it as well. Using normalized features help reduce the training time drastically.

## Description of deep NN
Our classifier NN model has input, output and 1 hidden layer respectively. The input layer has 8 nodes. The input layer has 4 nodes to match the dimensionality of the problem. The output layer has 3 nodes to match **One-Hot** encoding of the target labels. Below we present the main code block using PyTorch (for complete functional code see the jupyter notebook file).

```
batch_sz, D_in, H, D_out = 4, 4, 8, 3
learning_rate = 1e-4
# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
          torch.nn.Softmax(),
        )
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
Notice that we have used two different activation functions in hidden and output layers, namely **ReLU** and **Softmax** respectively. Further notice that we have used MSE (root-mean-squared-error) as loss function and **Adam** as optimization function. For more options on activation, loss and optimization functions look into https://pytorch.org/docs/stable/nn.html#

**Note**: While training the network we have used batch size of 4, to expedite the training process.
