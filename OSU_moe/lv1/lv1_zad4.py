"""Zadatak 1.4.4 Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ song.txt.
Potrebno je napraviti rjecnik koji kao klju ˇ ceve koristi sve razli ˇ cite rije ˇ ci koje se pojavljuju u ˇ
datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijec (klju ˇ c) pojavljuje u datoteci. ˇ
Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih."""

fhand = open ("song.txt") #otvoriti datoteku

text = fhand.read() #procitati

words = text.split() #razdvojit u rijeci

word_dict={}

for word in words:
    word = word.lower()

    if word not in word_dict:
        word_dict[word] = 1
    else:
        word_dict[word] += 1

print(word_dict) 