

#Install the following packages if not present

#!pip install contractions
#! pip install bs4 # in case you don't have it installed
# in case you don't have it installed

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz
#!pip install scikit-learn



import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')

#!python --version

#Python 3.7.13

"""## Read Data"""

df = pd.read_csv("data.tsv", 
                  usecols = ['star_rating','review_body'], sep='\t', on_bad_lines='skip')

"""## Keep Reviews and Ratings"""

df

"""Shuffling the rows"""

df = df.sample(frac=1).reset_index(drop=True)
df

"""Removing rows with null values"""

df.isnull().sum()

df.dropna(inplace=True)

df.isnull().sum()

df.star_rating.unique()

""" ## We select 20000 reviews randomly from each rating class.


"""

df.star_rating.value_counts()

df.dtypes

df['star_rating'] = df['star_rating'].astype(int)
df['review_body'] = df['review_body'].astype('string')

df.dtypes

df.star_rating.value_counts()

s1 = df[df.star_rating.eq(1)].sample(20000)
s2 = df[df.star_rating.eq(2)].sample(20000)
s3 = df[df.star_rating.eq(3)].sample(20000)
s4 = df[df.star_rating.eq(4)].sample(20000)
s5 = df[df.star_rating.eq(5)].sample(20000)

newdf = pd.concat([s1,s2,s3,s4,s5], ignore_index=True)
newdf

"""Shuffling the rows"""

newdf = newdf.sample(frac=1).reset_index(drop=True)
newdf

newdf.star_rating.value_counts()

newdf.isnull().sum()

# length= newdf["review_body"].str.len()
# length

# length.sum()/100000

before = newdf['review_body'].str.len().mean()

"""# Data Cleaning

Converting all the reviews to lower case
"""

newdf['review_body'] = newdf['review_body'].str.lower()

newdf

# regex = re.compile('[^a-zA-Z ]')
# #First parameter is the replacement, second parameter is your input string
# regex.sub('', 'ab3  d*E \n g')

#re.sub(' +', ' ',  '       T     he    ,    quick  bro      wn      fox     ')

def clean(row):
    

    soup = BeautifulSoup(row.review_body, "html.parser")

    #this extracts all the text from the document and removes html tags
    text1 = soup.get_text(' ')
    
    #removing any urls
    text2 = re.sub(r'http\S+', '', text1)
    #ftext = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)

    #performing contractions
    text3 = contractions.fix(text2)

    #removing non-alphabetic characters
    regex = re.compile('[^a-zA-Z ]')
    text4 = regex.sub('', text3)
    

    #removing extra white spaces
    text5 = re.sub(' +', ' ',  text4)

    row.review_body = text5

    return row

newdf = newdf.apply(clean, axis='columns')

newdf

newdf.isnull().sum()

after = newdf['review_body'].str.len().mean()

newdf.dropna(inplace=True)

print(before,',',after)

"""# Pre-processing

## remove the stop words
"""

stop = stopwords.words('english')

newdf['review_body'] = newdf['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#newdf['review_body'] = newdf['review_body'].apply(lambda x: [item for item in x.split() if item not in stop])

newdf

newdf.isnull().sum()

newdf.dropna(inplace=True)

#newdf['review_body'].str.len().mean()

"""## perform lemmatization  """

lemmatizer = WordNetLemmatizer()

#newdf['review_body'] = newdf['review_body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x]))
newdf['review_body'] = newdf['review_body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

newdf

later = newdf['review_body'].str.len().mean()

#newdf.isnull().sum()

print(after,',',later)
print('\n')

# c=0
# for i in newdf['review_body']:
#   if len(i.split())<3:
#     c+=1
# print(c)

"""# TF-IDF Feature Extraction"""

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newdf['review_body'])

X.shape

#X_features = pd.DataFrame(X.toarray())

"""# Train-Test Split"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, newdf['star_rating'], test_size=0.2)

y_train.value_counts()

y_test.value_counts()

"""Importing metrics to evaluate"""

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support

"""# Perceptron"""

from sklearn.linear_model import Perceptron

p = Perceptron()
p.fit(X_train,y_train)

predictions = p.predict(X_test)

#print(classification_report(y_test,predictions))

metrics = precision_recall_fscore_support(y_test, predictions)

"""Printing precision, recall, f1_score and their average"""

for i in range(5):
    print(metrics[0][i] , ',' , metrics[1][i] , ',' , metrics[2][i])
    
avg = np.mean(metrics[:3], axis=1)
print(avg[0], ',' ,avg[1], ',' ,avg[2])
print('\n')

"""# SVM"""

from sklearn.svm import LinearSVC
svm = LinearSVC()

svm.fit(X_train,y_train)

predictions = svm.predict(X_test)

#print(classification_report(y_test,predictions))

metrics = precision_recall_fscore_support(y_test, predictions)

"""Printing precision, recall, f1_score and their average"""

for i in range(5):
    print(metrics[0][i] , ',' , metrics[1][i] , ',' , metrics[2][i])
    
avg = np.mean(metrics[:3], axis=1)
print(avg[0], ',' ,avg[1], ',' ,avg[2])
print('\n')

"""# Logistic Regression"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)

predictions = lr.predict(X_test)

#print(classification_report(y_test,predictions))

metrics = precision_recall_fscore_support(y_test, predictions)

"""Printing precision, recall, f1_score and their average"""

for i in range(5):
    print(metrics[0][i] , ',' , metrics[1][i] , ',' , metrics[2][i])
    
avg = np.mean(metrics[:3], axis=1)
print(avg[0], ',' ,avg[1], ',' ,avg[2])
print('\n')

"""# Naive Bayes"""

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train, y_train)

predictions = nb.predict(X_test)

#print(classification_report(y_test,predictions))

metrics = precision_recall_fscore_support(y_test, predictions)

"""Printing precision, recall, f1_score and their average"""

for i in range(5):
    print(metrics[0][i] , ',' , metrics[1][i] , ',' , metrics[2][i])
    
avg = np.mean(metrics[:3], axis=1)
print(avg[0], ',' ,avg[1], ',' ,avg[2])

