import sys
import json
import string
import re
import math
import numpy as np

#path='/content/train-labeled.txt'
path=sys.argv[1]

uniquewords=set()
tagslist=[]
reviews=[]


with open(path, 'r', encoding='utf8') as f:

      sentencelist = f.readlines() 


      for sentence in sentencelist:
            words = sentence.split()
            tags=[]

            #id = words[0]
            tf = words[1]
            pn = words[2]

            #converting true/fake and pos/neg tags into 1 and -1

            if tf=='True':
              tags.append(1)
            else:
              tags.append(-1)
            
            if pn=="Pos":
              tags.append(1)
            else:
              tags.append(-1)

            tagslist.append(tags)

            sent=' '.join(words[3:])
            #print(sent)
            #sent = sent.translate(str.maketrans('', '', string.punctuation))
            #print(sent)
            sent = re.sub(r'[^A-Za-z ]+', '', sent)
            sent = sent.lower()
            sent = sent.split()

            reviews.append(sent)

            for i in sent:
                  uniquewords.add(i)

wordtonum={}
i=0
for word in uniquewords:
    wordtonum[word]=i
    i+=1
    
    
#getting counts for word in each sentence
l=len(uniquewords)
count=[]

for sent in reviews:
    feature = np.zeros(l)
    for word in sent:
        feature[wordtonum[word]]+=1
        #feature[word]+=1
    count.append(feature)

count = np.array(count)

#getting term frequencies
tf = count/count.sum(axis=1, keepdims=True)

doclen = len(tagslist)
#print(doclen)
idf = np.log(doclen/((count != 0).sum(0)+1))

features=[]
for i in range(doclen):
  features.append(tf[i] * idf)


tf=0
pn=1

epochs=25

c=0
l=len(uniquewords)

tfdict=np.zeros(l)
pndict=np.zeros(l)
tfbias=0
pnbias=0

avgtfdict=np.zeros(l)
avgpndict=np.zeros(l)
avgtfbias=0
avgpnbias=0

#tagslist=np.array(tagslist)
#features=np.array(features)

for e in range(epochs):
      #randomize = np.arange(len(features))
      #np.random.shuffle(randomize)
      #features = features[randomize]
      #tagslist = tagslist[randomize]
      for i in range(doclen):

            c+=1
            atf=0.0
            apn=0.0

            atf = features[i] * tfdict
            atf = np.sum(atf) + tfbias

            apn = features[i] * pndict
            apn = np.sum(apn) + pnbias

            if tagslist[i][tf] * atf <= 0:
                  tfdict = tfdict + features[i]*tagslist[i][tf]
                  tfbias = tfbias + tagslist[i][tf]

                  avgtfdict = avgtfdict + features[i]*tagslist[i][tf]*c
                  avgtfbias = avgtfbias + tagslist[i][tf]*c

            if tagslist[i][pn]*apn<=0:
                  pndict = pndict + features[i]*tagslist[i][pn]
                  pnbias = pnbias + tagslist[i][pn]

                  avgpndict = avgpndict + features[i]*tagslist[i][pn]*c
                  avgpnbias = avgpnbias + tagslist[i][pn]*c



#saving parameters for vanillla 
vanmasterlist=[]
vanmasterlist.append(wordtonum)
vanmasterlist.append(tfdict.tolist())
vanmasterlist.append(tfbias)
vanmasterlist.append(pndict.tolist())
vanmasterlist.append(pnbias)
#print(type(tfbias))
#print(tfbias)

with open('vanillamodel.txt', 'w', encoding='utf8') as jfile:
    json.dump(vanmasterlist, jfile, indent=1, ensure_ascii=False)


#saving parameters for average
avgpndict = avgpndict/c
avgpndict = pndict - avgpndict
avgpnbias = pnbias - avgpnbias/c

avgtfdict = avgtfdict/c
avgtfdict = tfdict - avgtfdict
avgtfbias = tfbias - (avgtfbias/c)

avgmasterlist=[]
avgmasterlist.append(wordtonum)
avgmasterlist.append(avgtfdict.tolist())
avgmasterlist.append(avgtfbias)
avgmasterlist.append(avgpndict.tolist())
avgmasterlist.append(avgpnbias)


with open('averagedmodel.txt', 'w', encoding='utf8') as jfile:
    json.dump(avgmasterlist, jfile, indent=1, ensure_ascii=False)
