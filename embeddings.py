

# https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
#from sentence_transformers import CrossEncoder
from chromadb import Documents, EmbeddingFunction, Embeddings
#import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np


EMBEDDINGS = [
    "ollama/nomic-embed-text:latest",
    "ollama/mxbai-embed-large",
    #"ollama/snowflake-arctic-embed",
    "ollama/all-minilm",
    "ollama/bge-m3",
    "ollama/bge-large",
    "ollama/paraphrase-multilingual",
    "ollama/snowflake-arctic-embed",    
    "seznam/Seznam/retromae-small-cs",
    "seznam/Seznam/dist-mpnet-paracrawl-cs-en",
    "seznam/Seznam/dist-mpnet-czeng-cs-en",
    "seznam/Seznam/simcse-retromae-small-cs",
    "seznam/Seznam/simcse-dist-mpnet-paracrawl-cs-en",
    "seznam/Seznam/simcse-dist-mpnet-czeng-cs-en",
    "seznam/Seznam/simcse-small-e-czech",
    "random"
    ]
EMBEDDING_DISTANCE_FUNCTIONS=["cosine", "l2", "ip"]


class SeznamEmbeddings(EmbeddingFunction):
    """https://github.com/seznam/czech-semantic-embedding-models"""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        #print(input[0])
        #print("===============================================")
        # texts=list(map(lambda d:d.text,input))
        # embed the documents somehow
        return self.embeddings(input)

    def embeddings(self, texts):
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            texts, max_length=128, padding=True, truncation=True, return_tensors="pt"
        )

        outputs = self.model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS
        return embeddings.detach().numpy()


def get_embedding_func(config_str):
    """  note that the embedding function returns an array of 1 embedding """

    retVal = None
    config = config_str.split("/", 1)
    #print("embedding:", str(config))
    if config[0] == "ollama":
        retVal = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings", model_name=config[1]
        )
    elif config[0] == "seznam":
        retVal = SeznamEmbeddings(config[1])
    elif config[0] == "random":        
        return lambda x: [np.random.rand(256)]
    else:
        raise "dunno embedding:" + str(config)

    # texts = ["Hello, world!", "How are you?"]
    # embeddings = retVal(texts)
    # print(embeddings)
    return retVal



def df_l2(a,b):
    x= np.linalg.norm(np.array(a)-np.array(b))
    return x/(1+x)


def df_cos(a,b):
    # using scipy
    #return spatial.distance.cosine(a, b)
    # using np
    b= np.transpose(b)
    return 1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_distance_func(name):
    if (name=='l2'):
         return df_l2
    elif (name=='cos'):
         return df_cos
    else:
         raise Exception("dunno distance func: "+name)
