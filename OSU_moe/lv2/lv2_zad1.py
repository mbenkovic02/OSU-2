"""Zadatak 2.4.1 Pomocu funkcija ´ numpy.array i 
matplotlib.pyplot pokušajte nacrtati sliku
2.3 u okviru skripte zadatak_1.py. Igrajte se sa slikom, promijenite boju linija, debljinu linije i
sl."""

import numpy
import matplotlib
import matplotlib.pyplot as plt

square = [(1, 1), (3, 1), (3, 2), (2, 2), (1, 1)]# moze se i odma razdvojit na x i y listu
x, y = zip(*square)

plt.plot(x ,y ,"b", linewidth=1, marker=".", markersize=5)
plt . axis ([0 ,4 ,0 , 4])
plt.show()