import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import os
import sklearn.metrics as met
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
import re
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data = pd.read_csv('gender-classifier.csv', encoding='latin-1')

data.drop (columns = ['_unit_id',
                      '_last_judgment_at',
                      'user_timezone',
                      'tweet_coord',
                      'tweet_count',
                      'tweet_created', 
                      'tweet_id',
                      'tweet_location',
                      'profileimage',
                      'created'], inplace = True)


data['gender'].value_counts()
drop_items_idx = data[data['gender'] == 'unknown'].index
data.drop (index = drop_items_idx, inplace = True)
data[data['profile_yn'] == 'no']['gender']
drop_items_idx = data[data['profile_yn'] == 'no'].index
data.drop (index = drop_items_idx, inplace = True)
data.drop (columns = ['profile_yn','profile_yn:confidence','profile_yn_gold'], inplace = True)
drop_items_idx = data[data['gender:confidence'] < 0.8].index
data.drop (index = drop_items_idx, inplace = True)

#sns.barplot (x = 'gender', y = 'fav_number',data = data, ci=None)
#plt.show()
#sns.barplot (x = 'gender', y = 'retweet_count',data = data, ci=None)
#plt.show()

twit_vocab = Counter()
for twit in data['text_tweet']:
    for word in twit.split(' '):
        twit_vocab[word] += 1

twit_vocab_reduced = Counter()
for w, c in twit_vocab.items():
    if not w in stop:
        twit_vocab_reduced[w]=c

def preprocessor(text):
    # Uklanjanje HTML tagova
    text = re.sub('<[^>]*>', '', text)
    # Cuvamo obicne smajlije
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Uklanjanje svi ne reci karaktera i njihovo dodavanje smajlijima
    # konvertovanje u lower case 
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))
    
    return text

pr = []
for rec in twit_vocab_reduced:
    pr.append(preprocessor(rec))

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tok = []
for rec in pr:
    tok.append(tokenizer_porter(rec))
print(tok)

flat_list = []
for sublist in tok:
    for item in sublist:
        flat_list.append(item)
print(flat_list)

with open('sve_reci.txt', 'w') as f:
    for item in flat_list:
        f.write("%s\n" % item)

#from seaborn import countplot
#from matplotlib.pyplot import figure, show
#figure()
#countplot(data['gender'],label="Gender")
#plt.title('Countplot')
#show()

#sns.barplot(x = 'gender', y = 'fav_number',data = data)
#show()
#sns.barplot(x = 'gender', y = 'retweet_count',data = data)
#show()
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

    #class_report = met.classification_report(y_train, y_pred, target_names=clf.classes_)
    #print("Izvestaj klasifikacije", class_report, sep="\n")


    print('Test skup')

    y_pred = clf.predict(x_test)

    cnf_matrix = met.confusion_matrix(y_test, y_pred)
    print("Matrica konfuzije", pd.DataFrame(cnf_matrix, index=clf.classes_, columns=clf.classes_), sep="\n")
    print("\n")

    accuracy = met.accuracy_score(y_test, y_pred)
    print("Preciznost", accuracy)
    print("\n")

    #class_report = met.classification_report(y_test, y_pred, target_names=clf.classes_)
    #print("Izvestaj klasifikacije", class_report, sep="\n")


encoder = LabelEncoder()
y = encoder.fit_transform(data['gender'])

X = data['text_tweet']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

n = range (1,100,10) #korak 10

results = []
for i in n:
    clf = Pipeline([('vect', tfidf),
                ('clf', RandomForestClassifier(n_estimators = i, random_state=0))])
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    results.append(accuracy_score(y_test, predictions))
plt.grid()
plt.scatter(n, results)
plt.xlabel('broj drveca u sumi')
plt.ylabel('preciznost')
plt.show()

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter,
                        preprocessor=preprocessor,stop_words = "english", max_features=500)
x = tfidf.fit_transform(X)
with open('reci_2.txt', 'w') as f:
    for item in tfidf.get_feature_names():
        f.write("%s, " % item)

df1 = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
df1['gender'] = y
export_csv = df1.to_csv(r'export_dataframe_novo_pretproc.csv', index = None, header=True)
print(df1)

clf = Pipeline([('vect', tfidf),
                ('clf', RandomForestClassifier(n_estimators = 90, random_state=0))])

clf.fit(X_train, y_train)
print('Trening skup')
predictions = clf.predict(X_train)
print('Accuracy:',accuracy_score(y_train,predictions))
print('Confusion matrix:\n',confusion_matrix(y_train,predictions))
print('Classification report:\n',classification_report(y_train,predictions))

print('Test skup')
predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
print('Confusion matrix:\n',confusion_matrix(y_test,predictions))
print('Classification report:\n',classification_report(y_test,predictions))

# poredjenje algoritama
x = ['MNB_1', 'DTC_1', 'KNN_1', 'KNN_PCA_1', 'NN_1','SVM_1','SVM_PCA_1', 'RFT_1', 'RFT_2', 'CART_2', 'C5.0_2',  'C5.0_PCA_2']
y = [0.5224,0.5241, 0.4693, 0.6179,0.4478, 0.3521, 0.5236, 0.3848, 0.5523, 0.5195, 0.5248, 0.5088]
plt.plot(x,y, 'o')
plt.xticks(x, rotation='vertical')
plt.show()
