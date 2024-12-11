from embeddings import *
import torch
import numpy as np
from scipy import spatial


def scorescore(expected,distance):
     """ score the distance """

     # check input
     if expected<0 or expected>1:
          raise Exception(""+str(expected))
     
     # handle rounding errors
     epsilon=0.1
     if distance<0 and distance>-epsilon:
          distance=0
     if distance>1 and distance<1+epsilon:
          distance=1
     if distance<0 or distance>1:
          raise Exception(""+str(distance))
     
     # l2 distance
     d=(expected-distance)*(expected-distance)

     # make a heavier cost at extremes
     f=(expected*2-1)*(expected*2-1)
     f=f/2+0.5

     return d*f



def compute_embedding_score(efname,dname,test_data,test_name):
    ef=get_embedding_func(efname)
    df=get_distance_func(dname)
    #print(embeddings[0])
    i=0
    ssacc=0
    embeddings=[]
    for j in range(0,len(test_data)):
        # print("test_data[",j,"][0]",test_data[j][0])
        # got exception 'Expected Embedings to be non-empty list or numpy array, got  ...'? try ollama pull model
        embedding=ef(test_data[j][0])
        embeddings.append(embedding[0])

    # normalize the diff, but only take care of the highest value
    diffs=[]
    for j in range(0,len(test_data)):
            diffs.append(df(embeddings[i],embeddings[j]))
    
    max = np.max(diffs)  
    diffns = diffs/max 

    print("| test | embedding | score | expected distance | actual distance norm | raw actual distance | text |")
    print("| -----| --------- | ----- | ----------------- | -------------------- | ------------------- | ---- |")
    
    for j in range(0,len(test_data)):       
            diff=diffs[j]
            diffn=diffns[j]
            ss=scorescore(test_data[j][1],diffn)
            ssacc+=ss
            print("|",test_name,"|",efname,"|",ss,"|",test_data[j][1],"|",diffn,"|",diff,"|",test_data[j][0])

    embeddingScore=(ssacc/len(test_data))
    print("#### avg score: | ",test_name," | ",embeddingScore," |",efname," |")





test_data_cz= [
("Zvyšte hmotnost vozu",0),
("Zvyšování vahu autu",0),
("Udělejte náklaďák těžší",0.1),
("Změňte hrubou tonáž vozidla",0.2),
("Změňte barvu vozidla",0.6),
("Dnes večer si dopřeji dobrý film a horkou čokoládu",1),
("My a našich 7 partnerů ukládáme a přistupujeme k osobním údajům, jako jsou údaje o prohlížení nebo jedinečné identifikátory, na vašem zařízení",1),
]

# first let s do some english tests
test_data_en = [    
    ("Increase the car weight",0),
    ("Increasing weight of the car",0),
    ("Make the truck more heavy",0.1), 
    ("Change the vehicle gross tonnage",0.2), 
    ("Change the colour of the vehicle",0.6), 
    ("Tonight I'll treat myself with a good movie and hot chocolate",1), # unrelated
    ("We and our 7 partners store and access personal data, like browsing data or unique identifiers, on your device",1), # unrelated
]


for embeddingName in EMBEDDINGS:
        compute_embedding_score(embeddingName,"cos",test_data_en,"en")    

for embeddingName in EMBEDDINGS:
        compute_embedding_score(embeddingName,"cos",test_data_cz,"cz")    



