
print('embeddings-1')
# https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq
__import__("pysqlite3")
print('embeddings-2')
import sys

print('embeddings-3')
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

print('embeddings-4')
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
print('embeddings-5')
#from sentence_transformers import CrossEncoder
from chromadb import Documents, EmbeddingFunction, Embeddings
print('embeddings-6')
#import torch
from transformers import AutoModel, AutoTokenizer
print('embeddings-7')
import numpy as np
print('embeddings-8')
#from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings



EMBEDDINGS = [
    "chromadb-ollama/nomic-embed-text:latest",
    "chromadb-ollama/mxbai-embed-large",
    #"ollama/snowflake-arctic-embed",
    "chromadb-ollama/all-minilm",
    "chromadb-ollama/bge-m3",
    "chromadb-ollama/bge-large",
    "chromadb-ollama/paraphrase-multilingual",
    "chromadb-ollama/snowflake-arctic-embed",    
    "langchain-ollama/nomic-embed-text:latest",
    "langchain-ollama/mxbai-embed-large",
    #"ollama/snowflake-arctic-embed",
    "langchain-ollama/all-minilm",
    "langchain-ollama/bge-m3",
    "langchain-ollama/bge-large",
    "langchain-ollama/paraphrase-multilingual",
    "langchain-ollama/snowflake-arctic-embed",    
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
    if config[0] == "chromadb-ollama":
        retVal = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings", model_name=config[1]
        )
    elif config[0] == "langchain-ollama":
        embed = OllamaEmbeddings(model=config[1])
        retVal=lambda t:[embed.embed_query(t)]
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


# TODO only cos is used, so just hardcode it
def df_l2(a,b):
    x= np.linalg.norm(np.array(a)-np.array(b))
    return x/(1+x)


def df_cos(a,b):
    # using scipy
    #return spatial.distance.cosine(a, b)
    # using np
    b= np.transpose(b)
    return 1.0-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_distance_func(name):
    if (name=='l2'):
         return df_l2
    elif (name=='cosine'):
         return df_cos
    else:
         raise Exception("dunno distance func: "+name)
