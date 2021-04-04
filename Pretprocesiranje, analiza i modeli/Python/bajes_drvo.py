'''
Klasifikacija samo na osnovu cistih reci iz teksta tvita,
eliminisane nepoznate vrednosti za pol (gender='unknown') 
i NAN vrednosti.
* stot reci izostavlje
* poterov stemer za korenovanje reci
* term matrica, uzeto 500 najfrekventnijih reci,
* inverzna term matrica
'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import stemrec as sp
from sklearn.feature_extraction.text import CountVectorizer, \
                                            TfidfVectorizer,  \
                                            TfidfTransformer 
from sklearn.naive_bayes import  MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored
import time
import os
from sklearn.feature_extraction import  DictVectorizer
import sklearn.metrics as met
from sklearn.neighbors import  KNeighborsClassifier
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer 

def class_info(clf, x_train, y_train, x_test, y_test, cv=False):

    start=time.time()
    clf.fit(x_train, y_train)

    end=time.time()
    print('Vreme', end-start)

    if cv:
        print('Najbolji parametri', clf.best_params_)

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) za %s" % (mean, std * 2, params))

    print('Trening skup')
    y_pred = clf.predict(x_train)

    cnf_matrix = met.confusion_matrix(y_train, y_pred)
    print("Matrica konfuzije", pd.DataFrame(cnf_matrix, index=clf.classes_, columns=clf.classes_), sep="\n")
    print("\n")

    accuracy = met.accuracy_score(y_train, y_pred)
    print("Preciznost", accuracy)
    print("\n")

    class_report = met.classification_report(y_train, y_pred, target_names=clf.classes_)
    print("Izvestaj klasifikacije", class_report, sep="\n")


    print('Test skup')

    y_pred = clf.predict(x_test)

    cnf_matrix = met.confusion_matrix(y_test, y_pred)
    print("Matrica konfuzije", pd.DataFrame(cnf_matrix, index=clf.classes_, columns=clf.classes_), sep="\n")
    print("\n")

    accuracy = met.accuracy_score(y_test, y_pred)
    print("Preciznost", accuracy)
    print("\n")

    class_report = met.classification_report(y_test, y_pred, target_names=clf.classes_)
    print("Izvestaj klasifikacije", class_report, sep="\n")

data = pd.read_csv("gender-classifier.csv",encoding="latin1")
print(data.shape)

drop_items_idx = data[data['profile_yn'] == 'no'].index
data.drop (index = drop_items_idx, inplace = True)
#print (data['profile_yn'].value_counts())
#print(data.shape)
print ('Podaci sa manje od 50% poverenja pri proceni profila: ', data[data['gender:confidence'] < 1].shape)
drop_items_idx = data[data['gender:confidence'] < 0.8].index
data.drop (index = drop_items_idx, inplace = True)

print(data['gender'].unique)
data = pd.concat([data.gender,data.text_tweet],axis=1)
drop_items_idx = data[data['gender'] == 'unknown'].index
data.drop(index = drop_items_idx, inplace = True)
data.dropna(axis=0,inplace=True)

#print(data.head())
#print(data.shape)
#print(data.head(10))
klase = data.gender

text_list = []
for text in data.text_tweet:
    text = re.sub("[^a-zA-Z]"," ",text)
    text = text.lower()
    text = text.split()
    text = " ".join(text)
    text_list.append(text)
   
p = sp.PorterStemmer()
text_lista= []
for line in text_list:
    output = ''
    word = ''
    for c in line:
        if c.isalpha():
            word += c.lower()
        else:
            if word:
                output += p.stem(word, 0,len(word)-1)
                word = ''
            output += c.lower()
    text_lista.append(output)
   
# Pretprocesiranje za dobijanje histograma najfrekventnijih reci po polovima korisnika   
#def cleaning(text):
    #text = re.sub("[^a-zA-Z]"," ",text)
    #text = text.lower()
    ##text = text.split()
    ##lemma = nltk.WordNetLemmatizer()
    ##text = [lemma.lemmatize(word) for word in text]
    ##text = " ".join(text)
    #output = ''
    #word = ''
    #for c in text:
        #if c.isalpha():
            #word += c.lower()
        #else:
            #if word:
                #output += p.stem(word, 0,len(word)-1)
                #word = ''
            #output += c.lower()
    #return output        
#data['Tweets'] = [cleaning(s) for s in data['text_tweet']]  
#from nltk.corpus import stopwords
#stop = set(stopwords.words('english'))
#data['Tweets'] = data['Tweets'].str.lower().str.split()
#data['Tweets'] = data['Tweets'].apply(lambda x : [item for item in x if item not in stop])  
##Histogrami reci po polu
#Male = data[data['gender'] == 'male']
#Female = data[data['gender'] == 'female']
#Brand = data[data['gender'] == 'brand']
#Male_Words = pd.Series(' '.join(Male['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]
#Female_Words = pd.Series(' '.join(Female['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]
#Brand_words = pd.Series(' '.join(Brand['Tweets'].astype(str)).lower().split(" ")).value_counts()[:10]

#Female_Words.plot(kind='bar',stacked=True, color='magenta')
#plt.show()

#Male_Words.plot(kind='bar',stacked=True, color='lightblue')
#plt.show()

#Brand_words.plot(kind='bar',stacked=True, color='gold')
#plt.show()

# maksimalan broj reci u term matrici, izdvajaju se one najfrekventnije
max_features = 500
# Pravljenje matrice broja pojavljivanja reci u dokumentu
#count_vectorizer = CountVectorizer(binary=True, max_features=max_features,stop_words = "english")
#x = count_vectorizer.fit_transform(text_lista)

# Pravljenje Tfidf matrice: 
count_vectorizer = TfidfVectorizer(lowercase=False, max_features=max_features,stop_words = "english")
x = count_vectorizer.fit_transform(text_lista)

#print('Atributi - reci')
#print(count_vectorizer.get_feature_names()) # vraca listu imena atributa
#print('Instance')
#print(x.toarray()) 

with open('reci.txt', 'w') as f:
    for item in count_vectorizer.get_feature_names():
        f.write("%s, " % item)


clf_mnb = MultinomialNB()
encoder = LabelEncoder()
y = encoder.fit_transform(klase)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, klase, train_size=0.7, stratify=y)

clf_mnb.fit(x_train, y_train)
class_info(clf_mnb, x_train, y_train, x_test, y_test)


print('Broj instanci po klasama')
print(pd.Series(clf_mnb.class_count_, index=clf_mnb.classes_))
print()
print('Broj pojavljivanja reci po klasama')
print(pd.DataFrame(clf_mnb.feature_count_, index=clf_mnb.classes_, columns=count_vectorizer.get_feature_names()))

#stampanje u fajl za SPSS:
#kol = count_vectorizer.get_feature_names()
#df = pd.DataFrame(x.toarray(), columns = kol)
#df['gender'] = y
#print(df)
#export_csv = df.to_csv(r'export_dataframe_500.csv', index = None, header=True)
#print('Multinomial: ') 
#print(clf_mnb.score(x_test, y_test))

kol = count_vectorizer.get_feature_names()
df = pd.DataFrame(x.toarray(), columns = kol)
df['gender'] = y
print(df)
export_csv = df.to_csv(r'export_dataframe_tfi.csv', index = None, header=True)

print(colored("DecisionTreeClassifier", "blue"))
parameters = [{'criterion': ['gini', 'entropy'],
               'max_depth':[15, 25, 5, 50],
               }]
clf_dt = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5)
class_info(clf_dt, x_train, y_train, x_test, y_test, cv=True)

#print(colored("KNeighborsClassifier", "blue"))
#parameters = [{'n_neighbors': range(3,10),
               #'p':[1, 2],
               #'weights': ['uniform', 'distance'],
               #}]

#clf_da = GridSearchCV(KNeighborsClassifier(), parameters, cv=5 )
#class_info(clf_da, x_train, y_train, x_test, y_test, cv=True)
