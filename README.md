# embedding-cmp - Naively compare embedding models for NLP

This is to check how good is the distance between 2 small sentences' embeddings. The goal is to make sure a language is supported by the
embedding.

Each of the test sentences is compared against the 1st sentence in the test: the cosinus is used as distance between embeddings and it is compared against the expected value (0:same meaning, 1:unrelated). So only one 'meaning' is tested.

Tests run against text in English and Czech.

The test runs against bi-encoders and cross-encoders, this is to check if it makes sense to use a cross-encoder at all.


⚠️ Note that the text used for tests is right now absurdely small: just 5 sentences !!! - so the results should be considered as an indication
if a language is supported rather than a leaderboad




### Install
```
python3 -m venv  .venv
. .venv/bin/activate
pip install -r requirements.txt

# pull the ollama models
echo "mxbai-embed-large snowflake-arctic-embed all-minilm bge-m3 bge-large paraphrase-multilingual" | xargs -n1 ollama pull 
```

### run the tests and sort avg scores (lower score == better)
```
python3 benchmark.py | tee out.md  && cat out.md | grep 'avg score' | cut -d: -f2- | sort -n

```


## Results:


detailed output:   [ out.md ](out.md)

Note on score:
Some models of Seznam have not been trained on English, interestingly it is still better than random
Some other models have not been trained in Czech (e.g.snowflake-arctic-embed)
As expected chromadb-ollama and langchain-ollama give exactly the same result
Average scores:

###  sorted by Language/Score

| lang |score | embedding | type |
| --- | --- | --- | --- |
 |  cz  |      0.0137  | chromadb-ollama/paraphrase-multilingual  | embedding |
 |  cz  |      0.0137  | langchain-ollama/paraphrase-multilingual  | embedding |
 |  cz  |      0.0138  | chromadb-ollama/bge-m3  | embedding |
 |  cz  |      0.0138  | langchain-ollama/bge-m3  | embedding |
 |  cz  |      0.0207  | seznam/Seznam/dist-mpnet-czeng-cs-en  | embedding |
 |  cz  |      0.0215  | seznam/Seznam/dist-mpnet-paracrawl-cs-en  | embedding |
 |  cz  |      0.0315  | seznam/Seznam/simcse-dist-mpnet-paracrawl-cs-en  | embedding |
 |  cz  |      0.0425  | seznam/Seznam/simcse-dist-mpnet-czeng-cs-en  | embedding |
 |  cz  |      0.0934  | seznam/Seznam/retromae-small-cs  | embedding |
 |  cz  |      0.1057  | seznam/Seznam/simcse-retromae-small-cs  | embedding |
 |  cz  |      0.1439  | chromadb-ollama/all-minilm  | embedding |
 |  cz  |      0.1439  | langchain-ollama/all-minilm  | embedding |
 |  cz  |      0.1596  | seznam/Seznam/simcse-small-e-czech  | embedding |
 |  cz  |      0.1621  | chromadb-ollama/bge-large  | embedding |
 |  cz  |      0.1621  | langchain-ollama/bge-large  | embedding |
 |  cz  |      0.1632  | chromadb-ollama/nomic-embed-text:latest  | embedding |
 |  cz  |      0.1632  | langchain-ollama/nomic-embed-text:latest  | embedding |
 |  cz  |      0.1633  | chromadb-ollama/mxbai-embed-large  | embedding |
 |  cz  |      0.1633  | langchain-ollama/mxbai-embed-large  | embedding |
 |  cz  |      0.2221  | random  | embedding |
 |  cz  |      0.2491  | chromadb-ollama/snowflake-arctic-embed  | embedding |
 |  cz  |      0.2491  | langchain-ollama/snowflake-arctic-embed  | embedding |
 |  en  |      0.0067  | chromadb-ollama/mxbai-embed-large  | embedding |
 |  en  |      0.0067  | langchain-ollama/mxbai-embed-large  | embedding |
 |  en  |      0.0071  | chromadb-ollama/bge-large  | embedding |
 |  en  |      0.0071  | langchain-ollama/bge-large  | embedding |
 |  en  |      0.0095  | seznam/Seznam/dist-mpnet-paracrawl-cs-en  | embedding |
 |  en  |      0.0098  | chromadb-ollama/bge-m3  | embedding |
 |  en  |      0.0098  | langchain-ollama/bge-m3  | embedding |
 |  en  |      0.0105  | seznam/Seznam/simcse-dist-mpnet-paracrawl-cs-en  | embedding |
 |  en  |      0.0116  | seznam/Seznam/dist-mpnet-czeng-cs-en  | embedding |
 |  en  |      0.0137  | chromadb-ollama/paraphrase-multilingual  | embedding |
 |  en  |      0.0137  | langchain-ollama/paraphrase-multilingual  | embedding |
 |  en  |      0.0144  | seznam/Seznam/simcse-dist-mpnet-czeng-cs-en  | embedding |
 |  en  |      0.0167  | chromadb-ollama/all-minilm  | embedding |
 |  en  |      0.0167  | langchain-ollama/all-minilm  | embedding |
 |  en  |      0.0175  | chromadb-ollama/snowflake-arctic-embed  | embedding |
 |  en  |      0.0175  | langchain-ollama/snowflake-arctic-embed  | embedding |
 |  en  |      0.0246  | chromadb-ollama/nomic-embed-text:latest  | embedding |
 |  en  |      0.0246  | langchain-ollama/nomic-embed-text:latest  | embedding |
 |  en  |      0.0475  | seznam/Seznam/simcse-retromae-small-cs  | embedding |
 |  en  |      0.0874  | seznam/Seznam/retromae-small-cs  | embedding |
 |  en  |      0.1260  | seznam/Seznam/simcse-small-e-czech  | embedding |
 |  en  |      0.2962  | random  | embedding |

 