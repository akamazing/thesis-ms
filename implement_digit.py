# Training Parameters
from include import *

# Training Parameters
mnist = input_data.read_data_sets("data/digit", one_hot=False)

# Network Parameters
num_classes = 10
num_images = 1000
from base_digit import *

yMatrix = []
predMatrix = []
accuracyMatrix = []
accMatrix = []

MT = ShiftX

for i in range(10):
    model_dir = "models/CNN/mnist/TrainedModel"+str(i)
    #model_dir = "models/Test"+str(i)
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)
    yMatrixTemp = []
    predMatrixTemp = []
    accuracyMatrixTemp = []

    for step in range(-50,50):
        with open("variables/Transformations/"+str(MT)+"/"+str(step+50), 'rb') as f:
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
 
fname = "variables/Output/Digit/CNN_"+str(MT)

if (os.path.exists(fname)):
        os.remove(fname)
with open(fname, 'wb') as f:
    pickle.dump([yMatrix, predMatrix, accuracyMatrix, accMatrix], f)


