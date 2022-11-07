import sys
import json
import string
import re
import math
import numpy as np

path = sys.argv[1]
#path = '/content/vanillamodel.txt'
#path = '/content/averagedmodel.txt'
with open(path, 'r', encoding='utf8') as jfile:
    j=json.load(jfile)

vocab = j[0]
tfdict = j[1]
tfbias = j[2]
pndict = j[3]
pnbias = j[4]

reviews=[]
ids=[]
#path='/content/dev-text.txt'
path=sys.argv[2]
with open(path, 'r', encoding='utf8') as f:

      sentencelist = f.readlines() 


      for sentence in sentencelist:
            words = sentence.split()

            id = words[0]

            sent=' '.join(words[1:])
            #print(sent)
            #sent = sent.translate(str.maketrans('', '', string.punctuation))
            #print(sent)
            sent = re.sub(r'[^A-Za-z ]+', '', sent)
            sent = sent.lower()
            sent = sent.split()

            reviews.append(sent)
            ids.append(id)

unk=0
l=len(vocab)
count=[]

for sent in reviews:
    feature = np.zeros(l)
    for word in sent:
        try:
          feature[vocab[word]]+=1
        except:
          unk+=1
        #feature[word]+=1
    count.append(feature)

count=np.array(count)

tf = count/count.sum(axis=1, keepdims=True)
doclen = len(ids)
idf = np.log(doclen/((count != 0).sum(0)+1))

features=[]
for i in range(doclen):
  features.append(tf[i] * idf)    


c=-1

with open('percepoutput.txt','w', encoding='utf8') as outf:
        for i in range(doclen):
              c+=1
              apn=0
              atf=0
              outputline = ids[c] + ' '

              atf = tfdict * features[i]
              atf = np.sum(atf) + tfbias

              apn = pndict * features[i]
              apn = np.sum(apn) + pnbias

              if atf<=0:
                  outputline+= 'Fake' +' '
              else:
                  outputline+= 'True' +' '

              if apn<=0:
                  outputline+= 'Neg' + '\n'
              else:
                  outputline+= 'Pos' +'\n'


              outf.write(outputline)
