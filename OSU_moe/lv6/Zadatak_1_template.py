import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("lv6/Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

""". Izradite algoritam KNN na skupu podataka za ucenje (uz ˇ K=5). Izracunajte to ˇ cnost ˇ
klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje. Usporedite ˇ
dobivene rezultate s rezultatima logisticke regresije. Što primje ˇ cujete vezano uz 
dobivenu ´ granicu odluke KNN modela?"""
#a)


KNN_model = KNeighborsClassifier ( n_neighbors = 5 )
KNN_model . fit ( X_train_n , y_train )

y_test_p_KNN = KNN_model . predict ( X_test )
y_train_p_KNN = KNN_model . predict ( X_train )

acc_test = accuracy_score(y_test, y_test_p_KNN)
acc_train = accuracy_score(y_train, y_train_p_KNN)


def do_KNN(neighbour: int):
    KNN_model = KNeighborsClassifier (n_neighbors=neighbour)
    KNN_model . fit ( X_train_n , y_train )

    y_test_p_KNN = KNN_model . predict ( X_test )
    y_train_p_KNN = KNN_model . predict ( X_train )

    print(f"KNN: {neighbour}")
    print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
    print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

    plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
    plt.tight_layout()
    plt.show()

do_KNN(1)
do_KNN(100)


KNN_model = KNeighborsClassifier(n_neighbors=7)
KNN_model.fit(X_train_n, y_train)
y_test_p_KNN = KNN_model.predict(X_test_n)
y_train_p_KNN = KNN_model.predict(X_train_n)

model = KNeighborsClassifier()
scores = cross_val_score(KNN_model, X_train, y_train, cv=5)
print(scores)

array = np.arange(1, 101)
param_grid = {'n_neighbors':array}
knn_gscv = GridSearchCV(model, param_grid , cv=5, scoring ='accuracy', n_jobs =-1)
knn_gscv.fit(X_train, y_train)
print(knn_gscv.best_params_)
print(knn_gscv.best_score_)
print(knn_gscv.cv_results_)
print(knn_gscv.best_params_)

"""model = KNeighborsClassifier(): This line creates an instance of the K-Nearest Neighbors (KNN) classifier with default parameters. It initializes the model.
scores = cross_val_score(KNN_model, X_train, y_train, cv=5): This line calculates the cross-validated scores of the KNN model using the cross_val_score function. 
It takes the KNN model (KNN_model), the input features (X_train), the target variable (y_train), and the number of folds for cross-validation (cv=5). 
It returns an array of scores for each fold.

array = np.arange(1, 101): This line creates an array of integers from 1 to 100 using NumPy's arange function.
 It will be used to define the range of values for the hyperparameter n_neighbors in the grid search.

param_grid = {'n_neighbors': array}: This line creates a parameter grid dictionary for the hyperparameter n_neighbors. 
The grid search will search for the best value of n_neighbors within the specified range (1 to 100).

knn_gscv = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1): This line initializes the grid search for hyperparameter tuning using the GridSearchCV class. 
It takes the model (model), the parameter grid (param_grid), the number of folds for cross-validation (cv=5), the scoring metric (scoring='accuracy'), and the number of jobs to run in parallel (n_jobs=-1).

knn_gscv.fit(X_train, y_train): This line fits the grid search model to the training data. 
It searches for the best hyperparameters (n_neighbors) using cross-validation on the training data.

print(knn_gscv.best_params_): This line prints the best hyperparameters found by the grid search.

print(knn_gscv.best_score_): This line prints the mean cross-validated accuracy score achieved by the best model.

print(knn_gscv.cv_results_): This line prints detailed results of the cross-validation performed by the grid search, 
including mean test scores and standard deviations for each hyperparameter combination."""