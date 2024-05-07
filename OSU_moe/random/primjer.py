import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
#from tensorflow import keras
#from keras import layers
#from keras.models import load_model

##################################################
# 1. zadatak
##################################################

# učitavanje dataseta
data = np.loadtxt('random\pima-indians-diabetes.csv', delimiter=',', skiprows=9)

# a)
print(f'Broj mjerenja: {len(data)}')

# b)
data_df = pd.DataFrame(data)
print(f'Broj dupliciranih: {data_df.duplicated().sum()}')
print(f'Broj izostalih: {data_df.isnull().sum()} ')
data_df = data_df.drop_duplicates()
data_df = data_df.dropna(axis=0) #trebalo je i izbacit sve 0 iz BMI
data = data[data[:,5]!=0.0] #######ovo je falilo, izbacivanje sve s 0.0 BMI
data_df = pd.DataFrame(data) #kreiranje ponovno data_df ali ovaj put s očišćenim podacima bez redaka s BMI 0.0
print(f'Broj preostalih: {len(data_df)}') 

# c)
plt.scatter(x=data[:, 7], y=data[:, 5])
plt.title('Odnos dobi i BMI')
plt.xlabel('Age(years)')
plt.ylabel('BMI(weight in kg/(height in m)^2)')
plt.show()
# BMI je pretežito izmedu 20 i 40 (kroz cijeli životni vijek, vidljivo je da je vise mjerenja odrađeno na mlađim ženama), uz nekoliko outliera kod kojih je BMI 0 (pogrešno očitanje) i preko 50

# d)
print(f'Minimalni BMI: {data_df[5].min()}')
print(f'Maksimalni BMI: {data_df[5].max()}')
print(f'Srednji BMI: {data_df[5].mean()}')

# e)
print(f'Minimalni BMI (dijabetes): {data_df[data_df[8]==1][5].min()}')
print(f'Maksimalni BMI (dijabetes): {data_df[data_df[8]==1][5].max()}')
print(f'Srednji BMI: (dijabetes) {data_df[data_df[8]==1][5].mean()}')

#[data_df[8]==1 znaci da gledamo tamo gdje je u 8. stupcu 1

print(f'Broj osoba s dijabetesom: {len(data_df[data_df[8]==1])}')

print(f'Minimalni BMI (nema dijabetes): {data_df[data_df[8]==0][5].min()}')
print(f'Maksimalni BMI (nema dijabetes): {data_df[data_df[8]==0][5].max()}')
print(f'Srednji BMI: (nema dijabetes) {data_df[data_df[8]==0][5].mean()}')

# Ljudi s dijabetesom u prosjeku imaju veći BMI, što je logično zbog posljedica same bolesti, maksimalni BMI osobe s dijabetesom je znatno veći nego one bez, a minimalni nije referentan jer je 0 u oba slučaja (nemoguće)

##################################################
# 2. zadatak
##################################################

# učitavanje dataseta
data_df = pd.DataFrame(data, columns=['num_pregnant', 'plasma', 'blood_pressure',
                       'triceps', 'insulin', 'BMI', 'diabetes_function', 'age', 'diabetes']) #koriste se ocisceni podaci za dataframe
X = data_df.drop(columns=['diabetes']).to_numpy()  #svi osim izlaznog
y = data_df['diabetes'].copy().to_numpy()      #izlazni-stupac koji govori ima li ili nema dijabetes

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# a)
logReg_model = LogisticRegression(max_iter=300)
logReg_model.fit(X_train, y_train)

# b)klasifikacija
y_predictions = logReg_model.predict(X_test)

# c)matrica zabune
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_predictions))
disp.plot()
plt.show()
# broj TN je 89, TP 36, FN 18 i FP 11, model često osobe koje imaju dijabetes proglasi da nemaju - greška, nedovoljno komentirano

# d)
print(f'Tocnost: {accuracy_score(y_test, y_predictions)}')
print(f'Preciznost: {precision_score(y_test, y_predictions)}')
print(f'Odziv: {recall_score(y_test, y_predictions)}')
# Model točno klasificira ljude kao dijabetičare ili ne s 81% točnost, udio stvarnih dijabetičara u skupu ljudi koje je model proglasio dijabetičarima je 76,5% (preciznost), a model od svih ljudi koji jesu dijabetičari točno predviđa da jesu njih 66,6% (odziv)
# greška, nedovoljno komentirano 

"""
Matrica zabune prikazuje da je model točno predvidio 89 osoba koje nemaju dijabetes (pravi negativi) i 36 osoba koje imaju dijabetes (pravi pozitivi). Međutim, 
model je pogrešno predvidio 18 osoba koje imaju dijabetes kao da ih nemaju (lažni negativi) i 11 osoba koje nemaju 
dijabetes kao da ga imaju (lažni pozitivi).

Točnost modela iznosi 0.783, što znači da je model točno klasificirao 78.3% ukupnih primjera.
Preciznost modela iznosi 0.674, što znači da je od svih primjera koje je model predvidio kao pozitivne, njih 67.4% zaista pozitivni. Odziv (senzitivnost) modela iznosi 0.633,
što znači da je model uspio prepoznati 63.3% svih stvarnih pozitivnih primjera
"""
##################################################
# 3. zadatak
##################################################
"""
# učitavanje podataka:
data_df = pd.DataFrame(data, columns=['num_pregnant', 'plasma', 'blood_pressure',
                       'triceps', 'insulin', 'BMI', 'diabetes_function', 'age', 'diabetes']) #koriste se ocisceni podaci za dataframe
X = data_df.drop(columns=['diabetes']).to_numpy()
y = data_df['diabetes'].copy().to_numpy()

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# a)izgradnja neuronske mreze
model = keras.Sequential()
model.add(layers.Input(shape=(8,)))  #zadano 8 varijabli
model.add(layers.Dense(units=12, activation="relu"))
model.add(layers.Dense(units=8, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.summary()  #ispis informacija o mrezi

# b)podesavanje procesa treniranja
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy", ])

# c)ucenje mreze
history = model.fit(X_train, y_train, batch_size=10,
                    epochs=150, validation_split=0.1)


# d)
model.save('Model/')

# e)evaluacija moreže
model = load_model('Model/')
score = model.evaluate(X_test, y_test, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')

# f)predikciju mreže,matrica zabune
y_predictions = model.predict(X_test)
y_predictions = np.around(y_predictions).astype(np.int32)
cm = confusion_matrix(y_test, y_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
# komentar u pdfu

Komentar: Kreirana mreža svakim novim pokretanjem ima drugačije parametre pa je zato ova matrica uvijek drugačija. Mreža ima rezultate podjednake onima kod logističke regresije, mreža često osobe koje 
imaju dijabetes proglasi da nemaju, broj TN je 82, TP 31, FN 23 I FP 18
"""


""" Zadatak 0.0.1 Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
 otkrivanja dijabetesa, pri ˇcemu se u devetom stupcu nalazi klasa 0 (nema dijabetes) ili klasa 1
 (ima dijabetes). Uˇcitajte dane podatke u obliku numpy polja data. Dodajte programski kod u
 skriptu pomo´ cu kojeg možete odgovoriti na sljede´ ca pitanja:
 a) Na temelju veliˇ cine numpy polja data, na koliko osoba su izvršena mjerenja?
 b) Postoje li izostale ili duplicirane vrijednosti u stupcima s mjerenjima dobi i indeksa tjelesne
 mase (BMI)? Obrišite ih ako postoje. Koliko je sada uzoraka mjerenja preostalo?
 c) Prikažite odnos dobi i indeksa tjelesne mase (BMI) osobe pomo´cu scatter dijagrama.
 Dodajte naziv dijagrama i nazive osi s pripadaju´cim mjernim jedinicama. Komentirajte
 odnos dobi i BMI prikazan dijagramom.
 d) Izraˇ cunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost indeksa tjelesne
 mase (BMI) u ovom podatkovnom skupu.
 e) Ponovite zadatak pod d), ali posebno za osobe kojima je dijagnosticiran dijabetes i za one
 kojima nije. Kolikom je broju ljudi dijagonosticiran dijabetes? Komentirajte dobivene
 vrijednosti.
 Zadatak 0.0.2 Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
 otkrivanja dijabetesa, pri ˇ cemu se u devetom stupcu nalazi izlazna veliˇ cina, predstavljena klasom
 0 (nema dijabetes) ili klasom 1 (ima dijabetes).
 Uˇcitajte dane podatke u obliku numpy polja data. Podijelite ih na ulazne podatke X i izlazne
2
 podatke y. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 80:20.
 Dodajte programski kod u skriptu pomo´ cu kojeg možete odgovoriti na sljede´ ca pitanja:
 a) Izgradite model logistiˇ cke regresije pomo´ cu scikit-learn biblioteke na temelju skupa poda
taka za uˇ cenje.
 b) Provedite klasifikaciju skupa podataka za testiranje pomo´ cu izgra¯ denog modela logistiˇ cke
 regresije.
 c) Izraˇ cunajte i prikažite matricu zabune na testnim podacima. Komentirajte dobivene rezul
tate.
 d) Izraˇcunajte toˇcnost, preciznost i odziv na skupu podataka za testiranje. Komentirajte
 dobivene rezultate.
 Zadatak 0.0.3 Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
 otkrivanja dijabetesa, pri ˇ cemu je prvih 8 stupaca ulazna veliˇcina, a u devetom stupcu se nalazi
 izlazna veliˇ cina: klasa 0 (nema dijabetes) ili klasa 1 (ima dijabetes).
 Uˇ citajte dane podatke. Podijelite ih na ulazne podatke X i izlazne podatke y. Podijelite podatke
 na skup za uˇ cenje i skup za testiranje modela u omjeru 80:20.
 a) Izgradite neuronsku mrežu sa sljede´ cim karakteristikama:- model oˇ cekuje ulazne podatke s 8 varijabli- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju- izlasni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
 Ispišite informacije o mreži u terminal.
 b) Podesite proces treniranja mreže sa sljede´ cim parametrima:- loss argument: cross entropy- optimizer: adam- metrika: accuracy.
 c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 150) i veliˇcinom
 batch-a 10.
 d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇ citanog modela.
 e) Izvršite evaluaciju mreže na testnom skupu podataka.
 f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
 podataka za testiranje. Komentirajte dobivene rezultate"""