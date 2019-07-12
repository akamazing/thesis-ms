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

# Load the Mnist test data
xTmp = mnist.test.images
yTmp = list(mnist.test.labels)
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

for i in range(10):
   
    model_dir = "models/CNN/mnist/TrainedModel"+str(i)
    #model_dir = "models/Test"+str(i)
    if (os.path.exists(model_dir)):
        shutil.rmtree(model_dir)
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    reloadData(Train)
    reloadData(Test)

    TrainModel (model, xTrain, yTrain, False)
    p,a = TestModel (model, xTest, yTest, False)
    print ("Step: "+str(i)+" Accuracy: "+str(a))    
