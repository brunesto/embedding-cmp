

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
    ("An increasing ratio will increase the weight of the car ",0),
    ("To make the truck more heavy, just add potatoes",0.1), 
    ("To change the vehicle gross tonnage, press the button",0.2), 
    ("Increase the car weight will increase its consumption",0,3),
    ("To change the colour of the vehicle, use a brush",0.6), 
    ("Automatic transfer can be achieved with a ",1), # unrelated
    ("Tonight I'll treat myself with a good movie and hot chocolate",1), # unrelated
    ("We and our 7 partners store and access personal data, like browsing data or unique identifiers, on your device",1), # unrelated
]



def compute_crossencoding_score(cname,test_data,test_name):
     print("## ",test_name,"|",cname)
     encoder_model = CrossEncoder(cname)

     pairs=[]
     for datum in test_data:
          pairs.append([test_data[0][0],datum[0]])
     diffs=encoder_model.predict(pairs)

     dump(cname, test_data, test_name, diffs,ftype="cross-encoder",invert_score=True)

for cross_encoder in CROSS_ENCODERS:
     compute_crossencoding_score(cross_encoder,test_data_en,"en")

for cross_encoder in CROSS_ENCODERS:
     compute_crossencoding_score(cross_encoder,test_data_cz,"cz")
