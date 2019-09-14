import pandas as pd
import warnings
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn import svm
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report, make_scorer,precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from functools import partial
from nltk import word_tokenize, pos_tag
from sklearn.linear_model import LogisticRegression
nltk.download('treebank')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

warnings.filterwarnings('ignore')

p=str(input("Enter obama/romney:"))
if(p=="obama"):
    cols = [' ','date','time','tweet','class','yourclass','hy']
    data = pd.read_excel("trainingObamaRomneytweets.xlsx", sheet_name='Obama',header=None,skiprows=2,names=cols)
    data_test=pd.read_csv('Obama_Test_dataset_NO_Label.csv',encoding = "ISO-8859-1")         #Dataframe object for test data
    output_file='Sri Vishnu_Maddala_Sai Aishwarya_Chavali_Obama.txt'

elif(p=="romney"):
    cols = [' ','date','time','tweet','class','yourclass','hy']
    data = pd.read_excel("trainingObamaRomneytweets.xlsx", sheet_name='Romney',header=None,skiprows=2,names=cols)
    data_test=pd.read_csv('Romney_Test_dataset_NO_Label.csv',encoding = "ISO-8859-1")         #Dataframe object for test data
    output_file='Sri Vishnu_Maddala_Sai Aishwarya_Chavali_Romney.txt'

else:
    print("Give the correct input")
    exit()


df=pd.DataFrame(data,columns=['tweet','class'])
df = df[(df['class'] != '!!!!')]
df=df[(df['class']!='irrevelant')]
df=df[(df['class']!='irrelevant')]
df=df[(df['class']!='IR')]

df = df[pd.notnull(df['class'])]
df = df[pd.notnull(df['tweet'])]
df= df[(df['class'].astype('int')!=2)]

model4 = svm.SVC(kernel='linear', C=1, gamma=1000)

tok = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()

pat1 = r'@[^\s]+'
pat2 = r'http\S+'
tag_re=re.compile(r'<.*?>')
combined_pat = r'|'.join((pat1, pat2))
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}

negation=["not","could","does","would","should"]
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
clean_tweet=[]
clean_tweet_test=[]
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def data_clean(text):
    f=[]
    f1=[]
    souped=tag_re.sub(" ",text)
    #souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    
    lower_case = clean.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    words = tok.tokenize(letters_only)
    f= [x for x in words if len(ps.stem(x))>2]              #Removes stop words and performs stemming
    for x in f:
        if(x in stop_words):
            if(x in negation):
                f1.append(x)
        else:
            f1.append(x)
    return (" ".join(f1)).strip()
    
df=df[(df['class'].astype(int)!=2)]

for i in df['tweet']:    
    clean_tweet.append(data_clean(i))
print("preprocessing done for training data")
    
for i in data_test['Tweet_text']:
    clean_tweet_test.append(data_clean(i))
print("preprocessing done for test data")
    

#For training data        
df['cleaned_tweet']=clean_tweet
df = df[pd.notnull(df['cleaned_tweet'])]
df['cleaned_tweet'].replace(" ",np.nan,inplace=True)
df=df.dropna()


#For test data
data_test['cleaned_tweet']=clean_tweet_test
data_test=data_test[pd.notnull(data_test['cleaned_tweet'])]
data_test['cleaned_tweet'].replace(" ",np.nan,inplace=True)
data_test=data_test.dropna()

tweet_id=data_test["Tweet_ID"]
X_test_tweet=np.array(data_test["cleaned_tweet"])



ngram_vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 3),stop_words="english")
ngram_vectorizer.fit(df['cleaned_tweet'])
encoder = preprocessing.LabelEncoder()

X = np.array(df['cleaned_tweet'])
y = np.array(df['class'])
y=y.astype('int')
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_y=y_train
    test_y=y_test

    #xtrain_count =  count_vect.transform(X_train)
    #xtest_count =  count_vect.transform(X_test)

    xtrain_tfidf =  ngram_vectorizer.transform(X_train)
    xtest_tfidf =  ngram_vectorizer.transform(X_test)

    model4.fit(xtrain_tfidf,train_y)
    y_pred = model4.predict(xtest_tfidf)
    #print(y_pred)
    #print("K nearest")


vect_test=ngram_vectorizer.transform(X_test_tweet)      #Transforming test data to ngram vectorizer
y_pred = model4.predict(vect_test)                      #predicting class lables for test data

i=0
with open(output_file, 'w') as f:
    for item in y_pred:
        f.write(str(tweet_id[i]) + "     "  +str(item)+"\n")
        i=i+1

print("Predicted values have been written to the file")
