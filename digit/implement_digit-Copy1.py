# Training Parameters
from base_mnist import *


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

xTmp = xTmp.reshape(xTmp.shape[0], 1, 28, 28)
# # convert from int to float
xTmp = xTmp.astype('float32')
xTest = []
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

for x in xTmp:
    xTest.append(np.transpose(x))
xTest = np.array(xTest)
yTest = yTmp
xTestBackup = xTest.copy()
yTestBackup = yTmp.copy()

# Load the Mnist train data
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

yMatrix = []
predMatrix = []
accuracyMatrix = []
accMatrix = []
for i in range(10):
    model_dir = "models/CNN/mnist/TrainedModel"+str(i)
    #model_dir = "models/Test"+str(i)
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)
    yMatrixTemp = []
    predMatrixTemp = []
    accuracyMatrixTemp = []

    for step in range(-50,50):
        reloadData(Test)
        angle = step
        #Transform (xTrain, yTrain, Rotate, angle)
        #TrainModel (model, xTrain, yTrain, False)
        Transform (xTest, yTest, Shear, angle)
        p,a = TestModel (model, xTest, yTest, False)
        yMatrixTemp.append(yTest)
        predMatrixTemp.append(p)
        accuracyMatrixTemp.append(a)
        print ("Step: "+str(step)+" Accuracy: "+str(a))
    accMatrixTemp = processResults(predMatrixTemp, yMatrixTemp)
    yMatrix.append(yMatrixTemp)
    predMatrix.append(predMatrixTemp)
    accuracyMatrix.append(accuracyMatrixTemp)
    accMatrix.append(accMatrixTemp)
    
with open("variables/digit/Shear/variables", 'wb') as f:
    pickle.dump([yMatrix, predMatrix, accuracyMatrix, accMatrix], f)


