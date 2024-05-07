import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

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
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


"""Učitavanje podataka: Koristimo pd.read_csv("penguins.csv") kako bismo učitali podatke iz CSV datoteke "penguins.csv" u DataFrame objekt.

Provjeravanje izostalih vrijednosti po stupcima: df.isnull().sum() provjerava koliko izostalih (NaN) vrijednosti ima u svakom stupcu. Metoda isnull() vraća DataFrame s boolean vrijednostima (True ako je vrijednost NaN, inače False), a zatim sum() računa broj True vrijednosti po stupcima.

Izbacivanje stupca "sex": df = df.drop(columns=['sex']) izbacuje stupac "sex" iz DataFrame objekta i ažurira DataFrame.

Uklanjanje redaka s izostalim vrijednostima: df.dropna(axis=0, inplace=True) uklanja sve retke koji sadrže barem jednu izostalu vrijednost (NaN) i ažurira DataFrame.

Kodiranje kategorijalne varijable "species": df['species'].replace({'Adelie' : 0, 'Chinstrap' : 1, 'Gentoo': 2}, inplace = True) zamjenjuje imena vrsta s numeričkim oznakama. Adelie je označen s 0, Chinstrap s 1 i Gentoo s 2.

Ispis informacija o DataFrame-u: print(df.info()) ispisuje informacije o DataFrame-u, uključujući broj redaka, broj stupaca i informacije o svakom stupcu (uključujući broj ne-null vrijednosti i tip podataka).

Definiranje ulaznih i izlaznih varijabli: input_variables = ['bill_length_mm', 'flipper_length_mm'] definira liste ulaznih i izlaznih varijabli.

Podjela podataka na trening i testni skup: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123) dijeli ulazne i izlazne varijable na trening i testni skup u omjeru 80:20. Random_state koristimo kako bismo osigurali reproducibilnost podjele."""



#a)

classes, counts_train=np.unique(y_train, return_counts=True)
classes, counts_test=np.unique(y_test, return_counts=True)
X_axis = np.arange(len(classes))
plt.bar(X_axis - 0.2, counts_train, 0.4, label = 'Train')
plt.bar(X_axis + 0.2, counts_test, 0.4, label = 'Test') 
plt.xticks(X_axis, ['Adelie(0)', 'Chinstrap(1)', 'Gentoo(2)'])
plt.xlabel("Penguins")
plt.ylabel("Counts")
plt.title("Number of each class of penguins, train and test data")
plt.legend()
plt.show()

"""Računanje broja pojavljivanja svake klase u trening i testnom skupu:
classes, counts_train=np.unique(y_train, return_counts=True): Ova linija koristi funkciju np.unique() za izračun jedinstvenih vrijednosti u y_train (tj. oznaka klasa) i vraća broj pojavljivanja svake klase u trening skupu.
classes, counts_test=np.unique(y_test, return_counts=True): Ova linija radi isto što i prethodna, ali za testni skup.

Priprema za crtanje grafa:
X_axis = np.arange(len(classes)): Stvara niz indeksa koji će se koristiti kao x-os u grafu.
plt.bar(X_axis - 0.2, counts_train, 0.4, label='Train'): Crtanje stupčastog dijagrama za broj pojavljivanja klasa u trening skupu.
plt.bar(X_axis + 0.2, counts_test, 0.4, label='Test'): Crtanje stupčastog dijagrama za broj pojavljivanja klasa u testnom skupu.

Postavljanje oznaka na x-os, y-os i naslov grafa:
plt.xticks(X_axis, ['Adelie(0)', 'Chinstrap(1)', 'Gentoo(2)']): Postavljanje oznaka na x-osi koje odgovaraju oznakama klasa.
plt.xlabel("Penguins"): Postavljanje oznake za x-os.
plt.ylabel("Counts"): Postavljanje oznake za y-os.
plt.title("Number of each class of penguins, train and test data"): Postavljanje naslova grafa.

Legenda:
plt.legend(): Dodavanje legende na graf, koja objašnjava koje su boje povezane s trening i testnim podacima."""

#b)

logisticRegression = LogisticRegression(max_iter=120)
logisticRegression.fit(X_train,y_train)

#c)

teta0 = logisticRegression.intercept_ 
coefs = logisticRegression.coef_ 
print('Teta0:')
print(teta0)
print('Parametri modela') 
print(coefs) 

#d)

plot_decision_regions(X_train, y_train, logisticRegression)

#e)

y_prediction = logisticRegression.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test,y_prediction))
disp.plot()
plt.title('Matrica zabune')
plt.show()
print(f'Tocnost: {accuracy_score(y_test,y_prediction)}')
print(classification_report(y_test,y_prediction))

