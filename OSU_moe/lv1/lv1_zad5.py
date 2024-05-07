"""Zadatak 1.4.5 Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ SMSSpamCollection.txt
[1]. Ova datoteka sadrži 5574 SMS poruka pri cemu su neke ozna ˇ cene kao ˇ spam, a neke kao ham.
Primjer dijela datoteke:
ham Yup next stop.
ham Ok lar... Joking wif u oni...
spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
12 Poglavlje 1. Uvod u programski jezik Python
a) Izracunajte koliki je prosje ˇ can broj rije ˇ ci u SMS porukama koje su tipa ham, a koliko je ˇ
prosjecan broj rije ˇ ci u porukama koje su tipa spam. ˇ
b) Koliko SMS poruka koje su tipa spam završava usklicnikom ?"""

fhand = open("SMSSpamCollection.txt")
lines = fhand.readlines()

total_words_ham = 0
total_ham_messages = 0
total_words_spam = 0
total_spam_messages = 0
exclamations_count = 0

for line in lines:
     label, message = line.split("\t")
     words = message.split()
     num_words = len(words)


     if label == "ham":
        total_words_ham += num_words
        total_ham_messages += 1
     elif label == "spam":
        total_words_spam += num_words
        total_spam_messages += 1

        if message.strip().endswith("!"):
            exclamations_count += 1

        
average_words_ham = total_words_ham / total_ham_messages
average_words_spam = total_words_spam / total_spam_messages

print("Average number of words in ham messages:", average_words_ham)
print("Average number of words in spam messages:", average_words_spam)
print("Number of spam messages ending with an exclamation mark:", exclamations_count)