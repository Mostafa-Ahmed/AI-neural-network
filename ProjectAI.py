import random

# This function to read datasets

def readFile(s):
    if s == "training":
        fh = open("Iris Data - training.csv", "r")
    if s == "testing":
        fh = open("Iris Data - testing.csv", "r")
    lines = fh.readlines()
    X=[]
    Y=[]
    for line in range(len(lines)):
        record = lines[line].split(",")
        X.append([])
        for i in range(4):
            X[line].append(float(record[i]))
        Y.append(record[4])

    for i in range(len(Y)):
        if Y[i]=="Iris-setosa":
            Y[i]=1
        if Y[i]=="Iris-virginica":
            Y[i]=-1
    return X,Y


def signumActivationFunction(value):

    if value > 0:
        return 1
    else:
        return -1


def PredictYhat(x,w,b):
    dotproduct = 0

    for i,j in zip(x,w):                  # Calculate dot product between each row of Feature X and 4 values of weight + bais
        dotproduct += i*j
    dotproduct+=b

    Yhat=signumActivationFunction(dotproduct)   # call signum function to get -1 or 1 for Y hat
    return  Yhat

def Perceptron(w,bais,learnrate):
    correctrecord=0
    x, y = readFile("training")
    for i in range(len(y)):
        Y_hat=PredictYhat((x[i]),w,bais)
        Error=y[i]-Y_hat
        if Y_hat != y[i]:                               # if there is error exist update weight
            w[0] = w[0] + (learnrate * Error * x[i][0])
            w[1] = w[1] + (learnrate * Error * x[i][1])
            w[2] = w[2] + (learnrate * Error * x[i][2])
            w[3] = w[3] + (learnrate * Error * x[i][3])
            bais = bais + (learnrate * Error)
        else :
            correctrecord=correctrecord+1
        #print(y[i], Y_hat)

    Accuracy= (correctrecord / len(y) ) * 100

    return  Accuracy ,w,bais

def TrainPerceptron():
    epochs=100
    learnrate = 0.001
    bais = 1
    w = []
    for i in range(0, 4):
        x = random.randint(-1, 1)
        w.append(x)
    for e in range(epochs+1):
        accuracy,w,bais=Perceptron(w,bais,learnrate)
        if e % 10 == 0:
            print("-----------Epoch",e,"------------\n","ACCURCY",accuracy)
    return w,bais

def TestPerceptron(w,bais):

    x,y=readFile("testing")
    correctrecord=0
    for i in range(len(y)):
        Y_hat = PredictYhat((x[i]), w, bais)

        if Y_hat == y[i]:
            correctrecord+=1

    Accuracy=(correctrecord/len(y))*100

    print( Accuracy )




#-----Main-------#
f,t=readFile("training")
w,b=TrainPerceptron()
print("weight",w)
print("bais",b)

print("Accuracy of TEST")
TestPerceptron(w,b)





