from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from timeit import default_timer as timer
from reader import Reader
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from joblib import Parallel, delayed
import io

from logger import config_log, print2

dataset_version = 'v2'

input_path = "H:\\mestrado\\aprendizadomaquina\\trabalho2\\datasetfakenews\\"

config_log(input_path)
reader = Reader(input_path, dataset_version)

def classify(model, modelname, basename, X_test, y_test, X_train, y_train):
    start = timer()
    #print2("rodando "+ modelname)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    end = timer()
    print2(modelname,basename, accuracy, (end-start), len(y_train), len(y_test), str(datetime.now()))
    dump(model, input_path+'/models/' + basename + '_' + modelname +'.joblib')
    #print2(y_train.count('0'), y_train.count('1'))
    #print2(y_test.count('0'), y_test.count('1'))

X_train, y_train = reader.read_train()
X_valid, y_valid = reader.read_validate()
X_test, y_test = reader.read_test()

#X_test, y_test = read_features("features100_test.tsv")

models = {}
#models["dummy"] = DummyClassifier(strategy='most_frequent')

print2("Inicio")

#for i in range(1, 6, 2):
#    for distance in ['euclidean', 'manhattan']:
#        models["knn_" + distance + "_" + str(i)] = KNeighborsClassifier(n_neighbors=i, weights='distance', metric=distance)
#models["tree"] = DecisionTreeClassifier(random_state=0)

for max_depth in range(25, 34, 5):
    models["randomDepth_" + str(max_depth)] = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, random_state=0)

for max_depth in range(35, 100, 5):
    models["randomDepth_" + str(max_depth)] = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, random_state=0)


models["gaussianNB"] = GaussianNB()

for j in [10, 20, 30, 40, 50, 60]:
    max_iter = 400
    models["mlpLayers_1_neurons"+str(j)+"_maxIter_"+str(max_iter)] = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=(j, ), random_state=0)
    models["mlpLayers_3_neurons" + str(j)+"_maxIter_"+str(max_iter)] = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=(j,j,j), random_state=0)
    models["mlpLayers_5_neurons" + str(j)+"_maxIter_"+str(max_iter)] = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=(j, j, j, j, j),random_state=0)
    models["mlpLayers_7_neurons" + str(j)+"_maxIter_"+str(max_iter)] = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=(j, j, j, j, j, j, j), random_state=0)

for i in range(-5, 6):
    models["linearSVC_" + str(pow(10.0, i))] = LinearSVC(C=float(pow(10.0, i)), random_state=0, max_iter = 5000)


for i in range(3, 6):
    models["SVC_" + str(pow(10.0, i))] = SVC(C=float(pow(10.0, i)), random_state=0, max_iter = 5000)

models["random_without_depth"] = RandomForestClassifier(random_state=0)

basename = dataset_version + "_train_"+str(len(y_train))+"_valid"+str(len(y_valid))
for key in models.keys():
    classify(models[key], key, basename, X_valid, y_valid)