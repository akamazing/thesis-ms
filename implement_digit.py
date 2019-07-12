# Training Parameters
from base_digit import *


# Training Parameters
mnist = input_data.read_data_sets("data/digit", one_hot=False)

# Network Parameters
num_classes = 10 # MNIST total classes (0-9 digits)
num_images = 1000

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
        with open("variables/Transformations/Shear/"+str(step+50), 'rb') as f:
            [xTest, yTest] = pickle.load(f)
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
    
with open("variables/Output/Digit/CNN", 'wb') as f:
    pickle.dump([yMatrix, predMatrix, accuracyMatrix, accMatrix], f)


