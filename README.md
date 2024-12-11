# embedding-cmp - Naively compare embedding models for NLP

This is to check how good is the distance between 2 small sentences' embeddings.
Tests run against text in English and Czech.
⚠️ Please note that the text used for benchmark is currently extremely small: just 5 sentences !!!


### Install
```
python3 -m venv  .venv
. .venv/bin/activate
pip install -r requirements.txt


ollama pull mxbai-embed-large snowflake-arctic-embed all-minilm bge-m3 bge-large paraphrase-multilingual
```

### run the tests and sort avg scores (lower score == better)
```
python3 benchmark.py | tee out.md  && cat out.md | grep 'avg score' | cut -d: -f2- | sort -n

```


## Results:



detailed output: [ out.md ](out.md)

Average scores:

###  Czech

| lang |score | embedding |
| --- | --- | --- |
  |  cz  |  0.013660907  | ollama/paraphrase-multilingual  |
 |  cz  |  0.013817386  | ollama/bge-m3  |
 |  cz  |  0.020721983  | seznam/Seznam/dist-mpnet-czeng-cs-en  |
 |  cz  |  0.02148078  | seznam/Seznam/dist-mpnet-paracrawl-cs-en  |
 |  cz  |  0.03154919  | seznam/Seznam/simcse-dist-mpnet-paracrawl-cs-en  |
 |  cz  |  0.04250988  | seznam/Seznam/simcse-dist-mpnet-czeng-cs-en  |
 |  cz  |  0.0933737  | seznam/Seznam/retromae-small-cs  |
 |  cz  |  0.105709486  | seznam/Seznam/simcse-retromae-small-cs  |
 |  cz  |  0.14391151  | ollama/all-minilm  |
 |  cz  |  0.15957502  | seznam/Seznam/simcse-small-e-czech  |
 |  cz  |  0.16212781  | ollama/bge-large  |
 |  cz  |  0.16318573  | ollama/nomic-embed-text:latest  |
 |  cz  |  0.16328217  | ollama/mxbai-embed-large  |
 |  cz  |  0.2319737827575458  | random  |
 |  cz  |  0.24907477  | ollama/snowflake-arctic-embed  |
 
snowflake-arctic-embed  beeing worst than random is an artifact of the test. It is as bad as random, because it does not support the Czech language

###  English

| lang |score | embedding |
| --- | --- | --- |
 |  en  |  0.0067326217  | ollama/mxbai-embed-large  |
 |  en  |  0.007063392  | ollama/bge-large  |
 |  en  |  0.009483382  | seznam/Seznam/dist-mpnet-paracrawl-cs-en  |
 |  en  |  0.009786382  | ollama/bge-m3  |
 |  en  |  0.010463563  | seznam/Seznam/simcse-dist-mpnet-paracrawl-cs-en  |
 |  en  |  0.011588998  | seznam/Seznam/dist-mpnet-czeng-cs-en  |
 |  en  |  0.01369808  | ollama/paraphrase-multilingual  |
 |  en  |  0.014435664  | seznam/Seznam/simcse-dist-mpnet-czeng-cs-en  |
 |  en  |  0.016708583  | ollama/all-minilm  |
 |  en  |  0.017452465  | ollama/snowflake-arctic-embed  |
 |  en  |  0.024561154  | ollama/nomic-embed-text:latest  |
 |  en  |  0.0474934  | seznam/Seznam/simcse-retromae-small-cs  |
 |  en  |  0.087387495  | seznam/Seznam/retromae-small-cs  |
 |  en  |  0.1259871  | seznam/Seznam/simcse-small-e-czech  |
 |  en  |  0.27706046534385825  | random  |

 I understand it as the last 3 models of Seznam have not been trained on English