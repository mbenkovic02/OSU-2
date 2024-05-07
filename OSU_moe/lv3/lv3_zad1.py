"""Zadatak 3.4.1 Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljede ´ ca pitanja: ´
a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili ˇ
duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke veli ˇ cine konvertirajte u tip ˇ
category.
b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal: ´
ime proizvoda¯ ca, model vozila i kolika je gradska potrošnja. ˇ
c) Koliko vozila ima velicinu motora izme ˇ du 2.5 i 3.5 L? Kolika je prosje ¯ cna C02 emisija ˇ
plinova za ova vozila?
d) Koliko mjerenja se odnosi na vozila proizvoda¯ ca Audi? Kolika je prosje ˇ cna emisija C02 ˇ
plinova automobila proizvoda¯ ca Audi koji imaju 4 cilindara? ˇ
e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na ˇ
broj cilindara?
f) Kolika je prosjecna gradska potrošnja u slu ˇ caju vozila koja koriste dizel, a kolika za vozila ˇ
koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva? ´
h) Koliko ima vozila ima rucni tip mjenja ˇ ca (bez obzira na broj brzina)? ˇ
i) Izracunajte korelaciju izme ˇ du numeri ¯ ckih veli ˇ cina. Komentirajte dobiveni rezultat."""


import pandas as pd
import matplotlib . pyplot as plt

data = pd . read_csv ( "data_C02_emission.csv")

#a)

print (f"Data frame ima {len(data)} mjerenja" )

for col in data.columns:
    print(f"{col} has a type of {data[col].dtype}")

print(data.info())

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].astype('category')

print(data.info())

print(f"Redovi s izostalim vrijednostima: {data.isnull().sum()}")
print(f"Duplicirane vrijednosti: {data.duplicated().sum()}")

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data = data . reset_index ( drop = True )


#b)

least_consumption = data.nsmallest(3, "Fuel Consumption City (L/100km)")
most_consumption = data.nlargest(3, "Fuel Consumption City (L/100km)")

print('Most consuming: ')
print(most_consumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']])
print('Least consuming: ')
print(least_consumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

#c)

specified_size = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print(f"Vozila sa kubikažom između 2.5 i 3.5L: {len(specified_size)}")
avg_co2= specified_size["CO2 Emissions (g/km)"].mean()
print(f"Prosjecni CO2 emissions za ta auta je: {avg_co2}")

#d)

audi_data = data[data["Make"] == "Audi"]
print(f"Vozila marke audi:{len(audi_data)}")

audi_data = audi_data[audi_data["Cylinders"]==4]
print(f"Prosjecni CO2 emissions za ta Audi sa 4 cilindra je: {audi_data['CO2 Emissions (g/km)'].mean()}")

#e)

cylinder_counts = data['Cylinders'].value_counts().sort_index()
print(cylinder_counts)

cylinder_emissions = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print("Cylinder emissions: ")
print(cylinder_emissions)

#f)

diesels = data[(data['Fuel Type'] == 'D')]
petrols = data[(data['Fuel Type'] == 'Z')]

print(f"Dizeli:\nProsjecno: {diesels['Fuel Consumption City (L/100km)'].mean()} - Medijalno: {diesels['Fuel Consumption City (L/100km)'].median()}")
print(f"Benzinci:\nProsjecno: {petrols['Fuel Consumption City (L/100km)'].mean()} - Medijalno: {petrols['Fuel Consumption City (L/100km)'].median()}")

#g)

vehicle = data[(data["Cylinders"]==4) & (data["Fuel Type"]=="D")]

print(f"Dizel s 4 cilindra s najvecom potrosnjom u gradu je:{vehicle.nlargest(1, 'Fuel Consumption City (L/100km)')}")

#h)

manual_vehicles = data[data["Transmission"].str[0] == "M"]
print(f"Broj vozila s rucnim mjenjacem:{len(manual_vehicles)}")

#i)

print(data.corr(numeric_only=True))

'''
Komentiranje zadnjeg zadatka:
Velicine imaju dosta veliki korelaciju. Npr. broj obujam motora i broj cilindara su oko 0.9, dok je potrosnja oko 0.8 sto ukazuje na veliku korelaciju.
Takodjer razlog zasto potrosnja u mpg ima veliku negativnu korelaciju je to sto je ta velicina obrnuta, odnosno, sto automobil vise trosi, broj je manji
Npr: automobil koji trosi 25 MPG trosi vise nego automobil koji trosi 45 MPG. Dakle, ta velicina je obrnuta L/100km te takodjer, zbog toga dobivamo negativnu
korelaciju. Sto je negativna korelacija blize -1 to je ona vise obrnuto proporcijalna, dok sto je blize 1, to je vise proporcijonalna. Vrijednosti oko 0
nemaju nikakvu korelaciju s velicinom.
'''