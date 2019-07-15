# Training Parameters
from include import *
from base_digit import *


# Training Parameters
mnist = input_data.read_data_sets("data/digit", one_hot=False)

xTest = mnist.test.images
yTest = mnist.test.labels

xTest = xTest.reshape(xTest.shape[0], 1, 28, 28)

#reloadData(Test)
yT = []
xT = []
for n in range(num_classes):
    train_filter = np.isin(yTest, [n])
    x, y = xTest[train_filter], yTest[train_filter]
    yT.append(list(y[:num_images]))
    xT.append(x[:num_images])
yTest = [i for l in yT for i in l]
xT = np.array(xT)
xTest = np.concatenate(xT)

for step in range(-50,50):
    print("MR: Rotate, Step: "+str(step+50))
    fout = "variables/Transformations/Rotate/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    x_test = Transform (xTest, yTest, Rotate, step)
    with open(fout, 'wb') as f:
        pickle.dump([x_test, yTest], f)


for step in range(-50,50):
    print("MR: Shear, Step: "+str(step+50))
    fout = "variables/Transformations/Shear/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    x_test = Transform (xTest, yTest, Shear, step)
    with open(fout, 'wb') as f:
        pickle.dump([x_test, yTest], f)
        
for step in range(-50,50):
    print("MR: ShiftX, Step: "+str(step+50))
    fout = "variables/Transformations/ShiftX/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    x_test = Transform (xTest, yTest, ShiftX, step)
    with open(fout, 'wb') as f:
        pickle.dump([x_test, yTest], f)


for step in range(-50,50):
    print("MR: ShiftY, Step: "+str(step+50))
    fout = "variables/Transformations/ShiftY/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    x_test = Transform (xTest, yTest, ShiftY, step)
    with open(fout, 'wb') as f:
        pickle.dump([x_test, yTest], f)
        
for step in range(-50,50):
    print("MR: Rotate, Step: "+str(step+50))
    fout = "variables/Transformations/Shade/"+str(step+50)
    if (os.path.exists(fout)):
        os.remove(fout)
    x_test = Transform (xTest, yTest, Shade, step)
    with open(fout, 'wb') as f:
        pickle.dump([x_test, yTest], f)