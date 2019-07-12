
# coding: utf-8

# In[1]:


from __future__ import division, print_function, absolute_import

# Import MNIST data
import os
import enum
import pickle
import logging
import shutil, sys                                                                                                                                                    
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#from keras.datasets import mnist
#from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.examples.tutorials.mnist import input_data

K.set_image_dim_ordering('th')
logging.getLogger("tensorflow").setLevel(logging.INFO)

mnist = input_data.read_data_sets("data/digit", one_hot=False)

Shade = "Shade"
Rotate = "Rotate"
Shear = "Shear"
ShiftX = "ShiftX"
ShiftY = "ShiftY"
Test = "Test"
Train = "Train"
# Training Parameters
mnist = input_data.read_data_sets("data/digit", one_hot=False)
# Training Parameters
learning_rate = 0.01
num_steps = 500
batch_size = 128

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit
num_images = 1000

# Load the Mnist test data
xTmp = mnist.test.images
yTmp = mnist.test.labels
yT = []
xT = []
for n in range(num_classes):
    train_filter = np.isin(yTmp, [n])
    x, y = xTmp[train_filter], yTmp[train_filter]
    yT.append(list(y[:num_images]))
    xT.append(x[:num_images])
yTmp = [i for l in yT for i in l]
xT = np.array(xT)
xTmp = np.concatenate(xT)
xTmp = xTmp.reshape(xTmp.shape[0], 1, 28, 28)
# # convert from int to float
xTmp = xTmp.astype('float32')
xTest = []
for x in xTmp:
    xTest.append(np.transpose(x))
xTest = np.array(xTest)
yTest = yTmp
xTestBackup = xTest.copy()
yTestBackup = yTmp.copy()

# Load the Mnist test data
xTmp = mnist.train.images
yTmp = list(mnist.train.labels)
xTmp = xTmp.reshape(xTmp.shape[0], 1, 28, 28)
# # convert from int to float
xTmp = xTmp.astype('float32')
xTrain = []
for x in xTmp:
    xTrain.append(np.transpose(x))
xTrain = np.array(xTrain)
yTrain = yTmp
xTrainBackup = xTrain.copy()
yTrainBackup = yTmp.copy()
# Create the neural network

def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters 
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)

        # Convolution Layer with 64 filters 
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)
    
    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)
    
    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 
        
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    
    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=pred_classes,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# In[ ]:


# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer

# Define the model function (following TF Estimator Template)
def nn_model_fn(features, labels, mode):
    
    # Build the neural network
    logits = neural_net(features)
    
    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)
    
    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 
        
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    
    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=pred_classes,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# In[ ]:


def reloadData(dataType):
    global xTestBackup
    global yTestBackup
    global xTrainBackup
    global yTrainBackup
    if dataType == "Test": # Reset test data
        global xTest
        global yTest
        xTest[:] = xTestBackup
        yTest[:] = yTestBackup
    elif dataType == "Train": # Reset training data
        global xTrain
        global yTrain
        xTrain[:] = xTrainBackup
        yTrain[:] = yTrainBackup
        


# In[ ]:
def TestModel(model, xData, yData, shuffle):
    #input_fn = tf.estimator.inputs.numpy_input_fn(
    #        x={'images': xData}, y= np.array(yData),
    #        batch_size=batch_size, shuffle=shuffle)
    #accuracy = model.evaluate(input_fn)

    # Prepare the input data
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': xData}, shuffle=shuffle)
    # Use the model to predict the images class
    preds = list(model.predict(input_fn))
    acc = 0.0
    for x in range(len(yData)):
        if (yData[x]==preds[x]):
            acc = acc+1
    acc = acc/len(yData)
    #return preds, accuracy['accuracy']
    return preds, acc


# In[ ]:


def TrainModel(model, xTrain, yTrain, shuffle):
    input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': xTrain}, y= np.array(yTrain),
            batch_size=batch_size, num_epochs=None, shuffle=shuffle)
    # Train the Model
    model.train(input_fn, steps=num_steps)


# In[ ]:


def Transform( xData, yData, tType, value ): # yData is not bein transformed
    
    n_images = len(xData)
    xTemp = []
    yTemp = []
    
    datagen = ImageDataGenerator()# fit parameters from data
    
    # tType = 1 for shade
    if tType == "Shade":
        Xnew = [[[[v-value if v-value>0.0 else 0.0 for v in n] for n in x[0]]] for x in xData]
        Xnew = np.array(Xnew)
        xData[:] = Xnew.astype('float32')
        return;

    # tType = 2 for rotation
    if tType == "Rotate":
        #datagen = ImageDataGenerator(rotation_range=value)# fit parameters from data
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'theta':value})
            xTemp.append(x)
            
    # tType = 3 for sheer
    if tType == "Shear":
        #datagen = ImageDataGenerator(shear_range=value)# fit parameters from data
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'shear':value})
            xTemp.append(x)
        
    # tType = 3 for shift
    if tType == "ShiftX":   
        #datagen = ImageDataGenerator(width_shift_range=value, height_shift_range=value)
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'tx':value})
            xTemp.append(x)
            
    # tType = 3 for shift
    if tType == "ShiftY":   
        #datagen = ImageDataGenerator(width_shift_range=value, height_shift_range=value)
        for i in range(n_images):
            x = datagen.apply_transform(xData[i], {'ty':value})
            xTemp.append(x)
        
    #datagen.fit(xData)
    # configure batch size and retrieve one batch of images
    #for xBatch, yBatch in datagen.flow(xData, yData, batch_size=n_images, shuffle=False):
        #xData[:] = xBatch
        #yData[:] = yBatch
        #break
    xData[:] = xTemp


# In[ ]:


def DisplayData( xTest, yTest ):
    for i in range(len(xTest)):
        plt.imshow(np.reshape(xTest[i], [28, 28]), cmap='gray')
        plt.show()
        print ("Image Label: ", yTest[i])


# In[ ]:


def OpenFile(MT):
    filename = str(MT)+".txt"
    fp = open(filename, 'w')
    return fp

def CloseFile(fp):
    fp.close()
    
#Shade = "Shade"
#Rotate = "Rotate"
#Sheer = "Sheer"
#Shift = "Shift"


# In[ ]:


def ShuffleMT (mode, fp, iteration):
    print ("Reloading training and test data")
    reloadData(Train)
    reloadData(Test)
    
    if (mode==Train):
        print ("Training model")
        TrainModel(xTrain, yTrain, True)
        
        print ("Evaluating Model")
        accuracy = EvaluateModel(xTest, yTest, False)
          
    elif (mode==Test):
        print ("Evaluating Model")
        accuracy = EvaluateModel(xTest, yTest, True)
        
    elif (mode=="Both"):
        print ("Training model")
        TrainModel(xTrain, yTrain, True)       
        print ("Evaluating Model")
        accuracy = EvaluateModel(xTest, yTest, True)

    print("Iteration: ", iteration, " Accuracy: ", accuracy)
    if (fp):
        fp.write(str(iteration)+"\t"+str(accuracy)+"\n")


# In[ ]:


def processResults (predMatrix, yMatrix):
    totalData = [0]*num_classes
    for y in yMatrix[0]:
        totalData[y] = totalData[y]+1
    aMatrix = []
    for i in range(len(predMatrix)):       
        accMatrix = [[0]*num_classes for x in range(num_classes)]
        for j in range(len(predMatrix[i])):
            accMatrix[yMatrix[i][j]][predMatrix[i][j]] += 1
            
        for x in range(len(accMatrix)):
            for y in range(len(accMatrix[x])):
                accMatrix[x][y] = round(accMatrix[x][y]/totalData[x], 3)
                
        aMatrix.append(accMatrix)
    return aMatrix


# In[ ]:





# In[ ]:




