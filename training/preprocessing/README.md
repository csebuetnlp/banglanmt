## Cleaning / Normalizing / Training Vocabularies
***The purpose of this extra cleaning on top of batch filtering is to maximize the amount of useful information in the bn-en dataset for a bilingual MT system. We do this by employing a variety of heuristics such as removing identical spans of foreign texts on both sides, applying transliteration when appropriate, thresholding allowed amount of foreign text in a sentence pair, etc. For more details, refer to the code.  Additionally, the script generates sentencepiece vocabulary files required for tokenizing the parallel corpora.*** 

### Usage

```bash
$ python preprocessor.py -h
usage: preprocessor.py [-h] --input_dir PATH --output_dir PATH [--normalize]
                      [--bn_vocab_size BN_VOCAB_SIZE]
                      [--en_vocab_size EN_VOCAB_SIZE]
                      [--bn_model_type BN_MODEL_TYPE]
                      [--en_model_type EN_MODEL_TYPE]
                      [--bn_coverage BN_COVERAGE] [--en_coverage EN_COVERAGE]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir PATH, -i PATH
                        Input directory
  --output_dir PATH, -o PATH
                        Output directory
  --normalize           Only normalize the files in input directory
  --bn_vocab_size BN_VOCAB_SIZE
                        bengali vocab size
  --en_vocab_size EN_VOCAB_SIZE
                        english vocab size
  --bn_model_type BN_MODEL_TYPE
                        bengali sentencepiece model type
  --en_model_type EN_MODEL_TYPE
                        english sentencepiece model type
  --bn_coverage BN_COVERAGE
                        bengali character coverage
  --en_coverage EN_COVERAGE
                        english character coverage
```

  * If the script is invoked with `--normalize`, it will only produce the normalized version of all .bn / .en files found in the `input_dir` in corresponding subdirectories of `output_dir`.
  * Otherwise, the script will recursively look for all filepairs (`X.bn`, `X.en`) inside `input_dir`, where `X` is any common file prefix, and produce the following files inside `output_dir`:
    
    * `combined.bn` / `combined.en`: filepairs obtained by cleaning all linepairs.
    * `bn.model`, `bn.vocab` / `en.model`, `en.vocab`: sentencepiece models


## Removing Evaluation pairs
***If you are training from scratch with new test / train datasets, you should remove all evaluation pairs (`validation` / `test`)  first from the training dataset to prevent data leakage.*** To do so, run `remove_evaluation_pairs.py`. 

**Make sure all datasets are normalized before running the script.** 

### Usage
```bash
$ python remove_evaluation_pairs.py -h
usage: remove_evaluation_pairs.py [-h] --input_dir PATH --output_dir PATH
                                  --src_lang SRC_LANG --tgt_lang TGT_LANG

optional arguments:
  -h, --help            show this help message and exit
  --input_dir PATH, -i PATH
                        Input directory
  --output_dir PATH, -o PATH
                        Output directory
  --src_lang SRC_LANG   Source language
  --tgt_lang TGT_LANG   Target language
```

* The input directory must be structured as mentioned [here](../). This script will remove all evaluation pairs from training pairs and write those to  `corpus.train.src_lang` / `corpus.train.tgt_lang` inside `output_dir`.
