from os import listdir
from os.path import isfile, join
from reader import Reader
from logger import config_log, print2
from joblib import dump, load
from timeit import default_timer as timer
from datetime import datetime

version = 'v2'
input_path = "H:/mestrado/aprendizadomaquina/trabalho2/datasetfakenews/"
model_path = input_path + 'models/best/'
print(model_path)

config_log(input_path)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    positive = '0'
    negative = '1'
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==positive:
           TP += 1
        if y_hat[i]==positive and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==negative:
           TN += 1
        if y_hat[i]==negative and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def testClassifier(modelfile, X_test, y_test):
    #print2('loading model', str(datetime.now()))
    model = load(model_path + modelfile)
    #print2('loaded', str(datetime.now()))
    start = timer()
    accuracy = model.score(X_test, y_test)
    y_predict = model.predict(X_test)
    TP, FP, TN, FN = perf_measure(y_test, y_predict)
    end = timer()
    print2(modelfile, accuracy, TP, FP, TN, FN, (end-start), str(datetime.now()))


reader = Reader(input_path, version)

modelfiles = [f for f in listdir(model_path) if isfile(join(model_path, f))]

print(modelfiles)

X_test, y_test = reader.read_test()

for modelfile in modelfiles:
    if modelfile.startswith(version):
        testClassifier(modelfile, X_test, y_test)
