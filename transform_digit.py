# Training Parameters
from base_digit import *


# Training Parameters
mnist = input_data.read_data_sets("data/digit", one_hot=False)
num_images = 1000
num_classes = 10

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


for step in range(-50,50):
    print("MR: Rotate, Step: "+str(step+50))
    fout = "variables/Transformations/Rotate/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    reloadData(Test)
    Transform (xTest, yTest, Rotate, step)
    with open(fout, 'wb') as f:
        pickle.dump([xTest, yTest], f)


for step in range(-50,50):
    print("MR: Shear, Step: "+str(step+50))
    fout = "variables/Transformations/Shear/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    reloadData(Test)
    Transform (xTest, yTest, Shear, step)
    with open(fout, 'wb') as f:
        pickle.dump([xTest, yTest], f)
        
for step in range(-50,50):
    print("MR: ShiftX, Step: "+str(step+50))
    fout = "variables/Transformations/ShiftX/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    reloadData(Test)
    Transform (xTest, yTest, ShiftX, step)
    with open(fout, 'wb') as f:
        pickle.dump([xTest, yTest], f)


for step in range(-50,50):
    print("MR: ShiftY, Step: "+str(step+50))
    fout = "variables/Transformations/ShiftY/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    reloadData(Test)
    Transform (xTest, yTest, ShiftY, step)
    with open(fout, 'wb') as f:
        pickle.dump([xTest, yTest], f)
        
for step in range(-50,50):
    print("MR: Rotate, Step: "+str(step+50))
    fout = "variables/Transformations/Shade/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    reloadData(Test)
    Transform (xTest, yTest, Shade, step)
    with open(fout, 'wb') as f:
        pickle.dump([xTest, yTest], f)