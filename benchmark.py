print('benchmark-1')
from embeddings import *
print('benchmark-2')
import torch
print('benchmark-3')
import numpy as np
print('benchmark-4')
from scipy import spatial
print('benchmark-5')
from sentence_transformers import CrossEncoder
print('benchmark-6')




CROSS_ENCODERS= (
            "cross-encoder/ms-marco-TinyBERT-L-2-v2",
            "cross-encoder/ms-marco-MiniLM-L-2-v2",
            "cross-encoder/ms-marco-MiniLM-L-4-v2",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            # Note: these 2 are not trained for sentence similarity, but for question answer
            #"cross-encoder/qnli-distilroberta-base",
            #"cross-encoder/qnli-electra-base",
            "cross-encoder/stsb-TinyBERT-L-4",
            "cross-encoder/stsb-distilroberta-base",
            "cross-encoder/stsb-roberta-base",
            "cross-encoder/stsb-roberta-large",
            "cross-encoder/quora-distilroberta-base",
           "cross-encoder/quora-roberta-base",
            "cross-encoder/quora-roberta-large",
          # ValueError: Converting from Tiktoken failed
          #   "cross-encoder/nli-deberta-v3-base",
          #   "cross-encoder/nli-deberta-base",
          #   "cross-encoder/nli-deberta-v3-xsmall",
          #   "cross-encoder/nli-deberta-v3-small",
          #   "cross-encoder/nli-roberta-base",
          #   "cross-encoder/nli-MiniLM2-L6-H768",
          #   "cross-encoder/nli-distilroberta-base",
             "BAAI/bge-reranker-base",
           #  "BAAI/bge-reranker-large",
          #   "BAAI/bge-reranker-v2-m3",
         #   "BAAI/bge-reranker-v2-gemma", # not initialized?
          #   "BAAI/bge-reranker-v2-minicpm-layerwise",
            "jinaai/jina-reranker-v1-tiny-en",
             "jinaai/jina-reranker-v1-turbo-en",
            "mixedbread-ai/mxbai-rerank-xsmall-v1",
             "mixedbread-ai/mxbai-rerank-base-v1",
          #   #"mixedbread-ai/mxbai-rerank-large-v1",
            "maidalun1020/bce-reranker-base_v1",
        )


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

def compute_crossencoding_score(cname,test_data,test_name):
     print("## ",test_name,"|",cname)
     encoder_model = CrossEncoder(cname)

     pairs=[]
     for datum in test_data:
          pairs.append([test_data[0][0],datum[0]])
     diffs=encoder_model.predict(pairs)

     dump(cname, test_data, test_name, diffs,ftype="cross-encoder",invert_score=True)

def s(x):return "{:10.4f}".format(x)

def dump(cname, test_data, test_name, diffs,ftype,invert_score=False):
    """ dump the score as a md table """
    # normalize the sentences diff according to intrisic meaning
    # this work because we expect the first match (equal sentences) to have the best value possible
    # and the worst value (for unrelated sentences) to be also present in the list of diffs
    diffns = normalize(diffs, invert_score) 


   
    print("| test | embedding | score | expected distance | actual distance norm | raw actual distance | text |")
    print("| -----| --------- | ----- | ----------------- | -------------------- | ------------------- | ---- |")
   
   
    ssacc=0
    for j in range(0,len(test_data)):       
           diff=diffs[j]
           diffn = diffns[j]
           ss=scorescore(test_data[j][1],diffn)
           ssacc+=ss
           print("|",test_name,"|",cname,"|",s(ss),"|",s(test_data[j][1]),"|",s(diffn),"|",s(diff),"|",test_data[j][0])

    avg_score=(ssacc/len(test_data))
    print("#### avg score: | ",test_name," | ",s(avg_score)," |",cname," |",ftype,"|")


def normalize(diffs, invert):
    """ linear normalization into 0...1 domain """
    dmin = np.min(np.array(diffs))  
    dmax = np.max(np.array(diffs))  
    drange=dmax-dmin

    diffns = diffs-dmin 
    diffns /= drange
    if (invert):
        diffns=1-diffns
    return diffns



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

    
    diffs=[]
    for j in range(0,len(test_data)):
            diffs.append(df(embeddings[i],embeddings[j]))
    
    dump(efname, test_data, test_name, diffs,ftype="embedding")




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

for cross_encoder in CROSS_ENCODERS:
     compute_crossencoding_score(cross_encoder,test_data_en,"en")

for cross_encoder in CROSS_ENCODERS:
     compute_crossencoding_score(cross_encoder,test_data_cz,"cz")


for embeddingName in EMBEDDINGS:
        compute_embedding_score(embeddingName,"cosine",test_data_en,"en")    

for embeddingName in EMBEDDINGS:
        compute_embedding_score(embeddingName,"cosine",test_data_cz,"cz")    



