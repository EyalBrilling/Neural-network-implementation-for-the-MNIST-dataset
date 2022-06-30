
import numpy as np
from scipy.special import expit as Sigmoid
from scipy.special import softmax as SoftMax
import sys




def NormalizeData(x_array):
    return x_array/255
    
def FormatFileDataToArray(file):
    
    data_array=np.loadtxt(file,max_rows=49999)
    return data_array


def InitiateWeights():
    """input layer to first hidden layer"""
    w1=np.random.uniform(low=-1,high=1,size=(128,784))
    b1=np.random.uniform(low=-1,high=1,size=(128,1))
    """first hidden layer to second hidden layer"""
    w2=np.random.uniform(low=-1,high=1,size=(64,128))
    b2=np.random.uniform(low=-1,high=1,size=(64,1))
    """second hidden layer to output layer"""
    w3=np.random.uniform(low=-1,high=1,size=(10,64))
    b3=np.random.uniform(low=-1,high=1,size=(10,1))
    return w1,b1,w2,b2,w3,b3

def NetWorkTagger(w1,b1,w2,b2,w3,b3,x_vector):
    z1 = np.dot(w1,x_vector)+b1
    h1=Sigmoid(z1)

    z2=np.dot(w2,h1)+b2
    h2= Sigmoid(z2)

    z3=np.dot(w3,h2)+b3
    h3=SoftMax(z3-max(z3))
    sumTest=sum(h3)
    tag=np.where(h3==max(h3))[0]
    tag=tag.item()
    return tag
    
def ForwardAndBackwardsPropagation(w1,b1,w2,b2,w3,b3,x_vector,y_vector,lr):
    """forward propagation"""
    z1 = np.dot(w1,x_vector)+b1
    h1=Sigmoid(z1)

    z2=np.dot(w2,h1)+b2
    h2= Sigmoid(z2)

    z3=np.dot(w3,h2)+b3
    h3=SoftMax(z3-max(z3))
    """backward propagation"""

    z3Min= h3 - y_vector # dC/dz3= dC/dh3 * dh3/dz3
    w3Min= w3 - lr*np.dot(z3Min,h2.T)  # dC/dw3= dC/dh3 * dh3/dz3 * dz3/ dw3 = z3Min * dz3/dw3
    b3Min= b3- lr * z3Min # dC/dd3= z3Min * dz3/db3
    
    z2Min=np.dot(w3.T, z3Min) * h2 * (1- h2)  # dC/dz2= z3Min * dz3/dh2 * dh2/dz2(sigmoid derative)
    w2Min= w2 - lr*np.dot(z2Min,h1.T) # dC/dw2= z2Min * dz2/dw2
    b2Min= b2- lr *z2Min # dC/db2= z2Min * dz2/db2

    z1Min= np.dot(w2.T,z2Min) * h1 * (1- h1)  # dC/dz1 = z2Min * dz2/dh1 * dh1/dz1(sigmoid derative)
    w1Min= w1 - lr*np.dot(z1Min,x_vector.T) # dC/dw1= z1Min * dz1/dw1
    b1Min= b1- lr * z1Min # dC/db1= z1Min * dz1/db1
    
    return w1Min,b1Min,w2Min,b2Min,w3Min,b3Min

def TrainWithNetwork(train_x,train_y):
    epoch = 15
    lr= 0.1
    w1,b1,w2,b2,w3,b3=InitiateWeights()
    for iteration in range(epoch):
        indexes = np.arange(train_x.shape[0])
        np.random.shuffle(indexes)
        for x_vector, y_correct_tag in zip(train_x[indexes], train_y[indexes]):

            x_vector=x_vector.reshape((784,1))
            y_vector=np.zeros((10,1))
            y_vector[int(y_correct_tag)][0]=1
            w1,b1,w2,b2,w3,b3=ForwardAndBackwardsPropagation(w1,b1,w2,b2,w3,b3,x_vector,y_vector,lr)
    return [w1,b1,w2,b2,w3,b3]

def TestNetwork(test_x,weightsWithBias):
    y_test_answers=[]
    for photo in test_x:
        photo=photo.reshape((784,1))
        y_test_answers.append(NetWorkTagger(*weightsWithBias,photo))
    return y_test_answers

def WriteAnswerToFile(answers_list,output_file):
    with open(output_file,'w') as file:
        for tag_answer in answers_list:
            file.write(f"{tag_answer}\n")

def CalculateSuccsessRate(algoTags,rightTags):
    right_answers=0
    wrong_answers=0
    for index,answer in enumerate(algoTags):
        if answer==rightTags[index]:
            right_answers+=1
        else:
            wrong_answers+=1
    return ((right_answers/(wrong_answers+right_answers))*100)


"""for testing only. requires importing "from sklearn.model_selection import train_test_split"""
def TrainAndTestArrayMakerFromTrainOnly(train_x_path,train_y_path):
    x_train_array= FormatFileDataToArray(train_x_path)
    y_train_array= FormatFileDataToArray(train_y_path)

    x_train_array,x_test_array,y_train_array,y_test_array =  train_test_split(x_train_array, y_train_array, test_size = 0.2)

    
    return (NormalizeData(x_train_array)),y_train_array,(NormalizeData(x_test_array)),y_test_array

train_x,train_y,test_x=sys.argv[1],sys.argv[2],sys.argv[3]



train_x=NormalizeData(np.loadtxt(train_x))
train_y=np.loadtxt(train_y)
test_x=NormalizeData(np.loadtxt(test_x))
"""
x_train_array,x_test_arra,y_train_array,y_test_array=TrainAndTestArrayMakerFromTrainOnly(train_x,train_y)
"""

final_weightsBias_list=TrainWithNetwork(train_x,train_y)
answers=TestNetwork(test_x,final_weightsBias_list)
WriteAnswerToFile(answers,"test_y")
"""
successRate=CalculateSuccsessRate(answers,y_test_array)
print({successRate})
"""