# FED
FED - Frame Event Detection
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FORMAS/FED/blob/main/notebook/colab-fed.ipynb)

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/andersonsacramento/fed)


# DESCRIPTION

FED is a closed domain event detector system for sentences in the Portuguese language. It detect events from sentences, i.e., event trigger identification and classification. The event types are based on the typology of the FrameNet project (BAKER; FILLMORE; LOWE, 1998). The models were trained on an enriched TimeBankPT (COSTA; BRANCO,2012) corpus.


Currently, in this Colab, 5 different trained models are available to execution: 0, 5, 25, 50, and 100 which respectively correspond to: 214, 137, 31, 13, and 5 event types.

The system outputs the event detections in the following Json format:
```json
[{
  "text":   "disse",
  "start":  58,
  "end":    63
  },
  ...
]
  
```

# Local Execution

## Prerequisites

1. Download and place the BERTimbau Base (SOUZA; NOGUEIRA;LOTUFO, 2020) model and vocabulary file:
    ```bash
    $ wget https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/bert-base-portuguese-cased_tensorflow_checkpoint.zip
	```
	```bash
	$ wget https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/vocab.txt
	```
	Then unzip and place it in the the models directory as follows:
	```
	├──models
	|      └── BERTimbau
	|               └── bert_config.json
	|               └── bert_model.ckpt.data-00000-of-00001
	|               └── bert_model.ckpt.index
	|               └── bert_model.ckpt.meta
	|               └── vocab.txt
	|
	|...
	```

2. Install the packages.
   ```bash
   $ pip install -r requirements.txt
   ```


# OPTIONS
    -h, --help                           Print this help text and exit
	--sentence  SENTENCE             Sentence string to detect events from
	--dir   INPUT-DIR OUTPUT-DIR     Detect events from files of input directory
		                         (one sentence per line) and write output json
					 files on output directory.
    --model  ID                          Identifier of models available: 0, 5, 25, 50 or 
	                                 100. The default model is 0


## EVENT DETECTION FROM FILES
The text files in the input directory are expected to have the format:

    * all text files end with the extension .txt
    * sentences are separated by newlines
	
```bash
$ python3 src/fed.py --dir /tmp/input-dir /tmp/output-dir
```
## EVENT DETECTION FROM A SENTENCE

```bash
$ python3 src/fed.py --sentence 'A Petrobras aumentou o preço da gasolina para 2,30 reais, disse o presidente.'
```
## How to cite this work

Peer-reviewed accepted paper:

10th Brazilian Conference on Intelligent Systems (BRACIS)

* Sacramento A. ; Souza M. . Joint Event Extraction with Contextualized Word Embeddings for the Portuguese 
Language.
