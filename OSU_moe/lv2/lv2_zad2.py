"""Zadatak 2.4.2 Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja ˇ data pri cemu je u ˇ
prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci´
stupac polja je masa u kg.
18 Poglavlje 2. Rad s bibliotekama Numpy i Matplotlib
a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja? ˇ
b) Prikažite odnos visine i mase osobe pomocu naredbe ´ matplotlib.pyplot.scatter.
c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom ˇ
podatkovnom skupu.
e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
ind = (data[:,0] == 1)"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv", delimiter=",", dtype="str")
data = data[1::]
data = np.array(data, np.float64)


# a)
num_individuals = len(data)
print("Number of individuals measured:", num_individuals)

# b)
height, weight = data[:,1], data[:, 2]
c = data[:, 0]
plt.scatter(height, weight, cmap="magma", alpha=0.5)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs. Weight')
plt.colorbar(label='Gender')
plt.show()

#c) 
height, weight = data[0::50, 1], data[0::50, 2]
plt.scatter(height, weight)
plt.show()


plt.scatter(height, weight, cmap="magma", alpha=0.5)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs. Weight')

plt.show()

#d)
mean = height.mean()
max = height.max()
min = height.min()

print(f"Max: {max}, min: {min}, mean: {mean}")

#e)

men = data[data[:,0]==1]
women= data[data[:,0]==0]


print(f"Males- Min:{men[:,1].min()}, Max:{men[:,1].max()}, Mean:{men[:,1].mean}")