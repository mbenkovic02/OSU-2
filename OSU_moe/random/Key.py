"""
 Zadatak 0.0.1 Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
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
 vrijednosti"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib . pyplot as plt
import pandas as pd

from sklearn . linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
from sklearn . metrics import accuracy_score
######################################################################################################################
data1 = np.loadtxt("random\pima-indians-diabetes.csv", delimiter=",", skiprows=9)

#a)

num_individuals = len(data1)
print("Number of individuals measured:", num_individuals)

#b)



data2 = pd. DataFrame(data1)

print(f"Redovi s izostalim vrijednostima: {data2.isnull().sum()}")
print(f"Duplicirane vrijednosti: {data2.duplicated().sum()}")

data2.drop_duplicates(inplace=True)
data2.dropna(axis=0,inplace=True)



data2 = data2 . reset_index ( drop = True )

data_numpy= data2.values

data_numpy = data_numpy[data_numpy[:,5]!=0.0]

print(f'Number of remaining rows after cleaning: {len(data_numpy)}')

data_pandas = pd. DataFrame(data_numpy)

print(f'Number of remaining rows after cleaning: {len(data_numpy)}') #provjera jel i ovaj oblik ociscen....

#c)

age, BMI = data_numpy[:,7], data_numpy[:, 5]
plt.scatter(age, BMI, alpha=0.5)
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Age vs BMI')
plt.show()

#Vecina ljudi se krece u rasponu od 20 do 40 za BMI index

#d)

print(f"Min vrijednost BMI:{data_pandas[5].min()}")
print(f"Max vrijednost BMI:{data_pandas[5].max()}")
print(f"Srednja vrijednost BMI:{data_pandas[5].mean()}")

#e)

diabetic = data_numpy[data_numpy[:,8]==1]
nondiabetic= data_numpy[data_numpy[:,8]==0]

print(f"Broj osoba s dijabetesom:{len(diabetic)}")
print(f"Diabetics- Min:{diabetic[:,5].min()}, Max:{diabetic[:,5].max()}, Mean:{diabetic[:,5].mean()}")
print(f"Nondiabetics- Min:{nondiabetic[:,5].min()}, Max:{nondiabetic[:,5].max()}, Mean:{nondiabetic[:,5].mean()}")

# Ljudi s dijabetesom u prosjeku imaju veći BMI, što je logično zbog posljedica same bolesti, maksimalni BMI osobe s dijabetesom je znatno veći nego one bez

#############################################################################################################################################################################

data_df = pd.DataFrame(data_numpy, columns=['num_pregnant', 'plasma', 'blood_pressure', 'triceps', 'insulin', 'BMI', 'diabetes_function', 'age', 'diabetes']) #koriste se ocisceni podaci za dataframe
#za assignanje imena stupcima dataframea
#bitno dati numpy jer ako dam pandas ima nan vrijednosti predane X-u

input_variables = ['num_pregnant', 'plasma', 'blood_pressure', 'triceps', 'insulin', 'BMI', 'diabetes_function', 'age']
output = 'diabetes'

X = data_df[input_variables]
y = data_df[output]

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =1 )


#If you are using libraries like scikit-learn for machine learning tasks,
#both NumPy arrays and Pandas DataFrames are supported. 
#However, scikit-learn typically expects input data in the form of NumPy 
#arrays. Therefore, you might find it more convenient to convert your 
#Pandas DataFrame to a NumPy array before splitting if you're using 
#scikit-learn.


LogRegression_model = LogisticRegression (max_iter=300)
LogRegression_model.fit( X_train , y_train )


#klasifikacija
y_test_p = LogRegression_model.predict(X_test)



# matrica zabune
cm = confusion_matrix ( y_test , y_test_p )
print (" Matrica zabune : " , cm )
disp = ConfusionMatrixDisplay ( confusion_matrix ( y_test , y_test_p ) )
disp . plot ()
plt . show ()

#Matrica zabune govori da je od 87 ljudi od 110 ljudi sa dijabetesom uspjesno predvidjeno 87 a 22 su lazno negativni. Za ljude bez dijabetesa 31 je tocno a 12 je lazno pozitivno

#tocnost preciznost i odziv

print ("Tocnost:" , accuracy_score ( y_test , y_test_p ) )
print(f'Preciznost: {precision_score(y_test , y_test_p)}')
print(f'Odziv: {recall_score(y_test , y_test_p)}')


"""
Matrica zabune prikazuje da je model točno predvidio 89 osoba koje nemaju dijabetes (pravi negativi) i 36 osoba koje imaju dijabetes (pravi pozitivi). Međutim, 
model je pogrešno predvidio 18 osoba koje imaju dijabetes kao da ih nemaju (lažni negativi) i 11 osoba koje nemaju 
dijabetes kao da ga imaju (lažni pozitivi).

Točnost modela iznosi 0.783, što znači da je model točno klasificirao 78.3% ukupnih primjera.
Preciznost modela iznosi 0.674, što znači da je od svih primjera koje je model predvidio kao pozitivne, njih 67.4% zaista pozitivni. Odziv (senzitivnost) modela iznosi 0.633,
što znači da je model uspio prepoznati 63.3% svih stvarnih pozitivnih primjera
"""