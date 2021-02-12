## Setup
* Install all dependecies mentioned [here](https://github.com/csebuetnlp/banglanmt).
* download models: `bash ./install_models.sh`
* setup necessary tools: `bash ./install_external_tools.sh`

## Usage
* setup environment variable before running.
  ```bash
  # inside this directory
  $ export LASER=$(pwd)
  ```
* Batch filtering options
  ```bash
  $ python3 scoring_pipeline.py -h
  usage: scoring_pipeline.py [-h] --input_dir PATH --output_dir PATH --src_lang
                           SRC_LANG --tgt_lang TGT_LANG [--thresh THRESH]
                           [--batch_size BATCH_SIZE] [--cpu]

  optional arguments:
    -h, --help            show this help message and exit
    --input_dir PATH, -i PATH
                          Input directory
    --output_dir PATH, -o PATH
                          Output directory
    --src_lang SRC_LANG   Source language
    --tgt_lang TGT_LANG   Target language
    --thresh THRESH       threshold
    --batch_size BATCH_SIZE
                          batch size

  ```
  *  ***The script will recursively look for all filepairs `(X.src_lang, X.tgt_lang)` inside `input_dir`, where `X` is any common file prefix, and produce the following output files within the corresponding subdirectories of `output_dir`***
  
     * `X.merged.tsv`: Output linepairs with their similarity score
     * `X.passed.src_lang` / `X.passed.tgt_lang`: Linepairs that have similarity scores greater than given `thresh`
     * `X.failed.src_lang` / `X.failed.tgt_lang`: Linepairs that have similarity scores less than given `thresh`
