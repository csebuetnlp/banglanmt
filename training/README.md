# Preprocessing

If you want to,
*  build a new bn-en training dataset from a noisy parallel corpora (by filtering / cleaning some pairs based on our heuristics) with corresponding vocabulary models or
*  normalize a new dataset before evaluating on the model or
*  remove all evaluation pairs from training pairs for a new set of training / test datasets 

refer to [here](preprocessing/).

# Training

### Setup
* Install all dependecies mentioned in the home page first.
* ```bash
  $ cd seq2seq/
  $ pip install -r requirements.txt
  $ pip install ./

### Usage

```bash
$ cd seq2seq/pipeline/
$ python pipeline.py -h
usage: pipeline.py [-h] --input_dir PATH --output_dir PATH --src_lang SRC_LANG
                   --tgt_lang TGT_LANG
                   [--validation_samples VALIDATION_SAMPLES]
                   [--src_seq_length SRC_SEQ_LENGTH]
                   [--tgt_seq_length TGT_SEQ_LENGTH]
                   [--model_prefix MODEL_PREFIX] [--eval_model PATH]
                   [--train_steps TRAIN_STEPS]
                   [--train_batch_size TRAIN_BATCH_SIZE]
                   [--eval_batch_size EVAL_BATCH_SIZE]
                   [--gradient_accum GRADIENT_ACCUM]
                   [--warmup_steps WARMUP_STEPS]
                   [--learning_rate LEARNING_RATE] [--layers LAYERS]
                   [--rnn_size RNN_SIZE] [--word_vec_size WORD_VEC_SIZE]
                   [--transformer_ff TRANSFORMER_FF] [--heads HEADS]
                   [--valid_steps VALID_STEPS]
                   [--save_checkpoint_steps SAVE_CHECKPOINT_STEPS]
                   [--average_last AVERAGE_LAST] [--world_size WORLD_SIZE]
                   [--gpu_ranks [GPU_RANKS [GPU_RANKS ...]]]
                   [--train_from TRAIN_FROM] [--do_train] [--do_eval]
                   [--do_preprocess] [--nbest NBEST] [--alpha ALPHA]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir PATH, -i PATH
                        Input directory
  --output_dir PATH, -o PATH
                        Output directory
  --src_lang SRC_LANG   Source language
  --tgt_lang TGT_LANG   Target language
  --validation_samples VALIDATION_SAMPLES
                        no. of validation samples to take out from train
                        dataset when no validation data is present
  --src_seq_length SRC_SEQ_LENGTH
                        maximum source sequence length
  --tgt_seq_length TGT_SEQ_LENGTH
                        maximum target sequence length
  --model_prefix MODEL_PREFIX
                        Prefix of the model to save
  --eval_model PATH     Path to the specific model to evaluate
  --train_steps TRAIN_STEPS
                        no of training steps
  --train_batch_size TRAIN_BATCH_SIZE
                        training batch size (in tokens)
  --eval_batch_size EVAL_BATCH_SIZE
                        evaluation batch size (in sentences)
  --gradient_accum GRADIENT_ACCUM
                        gradient accum
  --warmup_steps WARMUP_STEPS
                        warmup steps
  --learning_rate LEARNING_RATE
                        learning rate
  --layers LAYERS       layers
  --rnn_size RNN_SIZE   rnn size
  --word_vec_size WORD_VEC_SIZE
                        word vector size
  --transformer_ff TRANSFORMER_FF
                        transformer feed forward size
  --heads HEADS         no of heads
  --valid_steps VALID_STEPS
                        validation interval
  --save_checkpoint_steps SAVE_CHECKPOINT_STEPS
                        model saving interval
  --average_last AVERAGE_LAST
                        average last X models
  --world_size WORLD_SIZE
                        world size
  --gpu_ranks [GPU_RANKS [GPU_RANKS ...]]
                        gpu ranks
  --train_from TRAIN_FROM
                        start training from this checkpoint
  --do_train            Run training
  --do_eval             Run evaluation
  --do_preprocess       Run the preprocessor to create binary files. Delete
                        previous files if theres any.
  --nbest NBEST         sentencepiece nbest size
  --alpha ALPHA         sentencepiece alpha
```

*  ***Sample `input_dir` structure for bn2en training:***

    ```bash
    input_dir/
    |---> data/
    |    |---> corpus.train.bn
    |    |---> corpus.train.en
    |    |---> RisingNews.valid.bn
    |    |---> RisingNews.valid.en
    |    |---> RisingNews.test.bn
    |    |---> RisingNews.test.en
    |    |---> sipc.test.bn
    |    |---> sipc.test.en.0
    |    |---> sipc.test.en.1
    |    ...
    |---> vocab/
    |    |---> bn.model
    |    |---> en.model
    ```
     * Input data files inside the `data/` subdirectory must have the following format: **`X.type.lang(.count)`**, where `X` is any common file prefix, `type` is one of `{train, valid, test}` and `count` is an optional integer (**only applicable for the `target_lang`, when there are multiple reference files**). There can be multiple `.train.`/`.valid.` filepairs. In absence of `.valid.` files, `validation_samples` no of example pairs will be randomly sampled from the training files during `training`.
     * The `vocab` subdirectory must hold two sentencepiece vocabulary models formatted as `src_lang.model` and `tgt_lang.model`
 
* ***After training / evaluation, the `output_dir` will have the following subdirectories with these contents.***
   * `Models`:  All the saved models
   * `Reports`:  **BLEU and SACREBLEU** scores on the validation files for all saved models with the given `model_prefix`, and the scores on the test files for the given `eval_model` (if the corresponding reference files are present)
   * `Outputs`: Detokenized model predictions.
   * `data`: Merged training files after applying subword regularization.
  * `Preprocessed`: Training and validation data shards
   

***To reproduce our results on an AWS  p3.8xlarge ec2 instance, equipped with 4 Tesla V100 GPUs, run the script with the default hyperparameters.*** For example, for bn2en training,
```bash
$ export CUDA_VISIBLE_DEVICES=0,1,2,3
# for training
$ python pipeline.py --src_lang bn --tgt_lang en -i inputFolder/ -o outputFolder/ --model_prefix bn2en --do_train --do_preprocess --do_eval
# for evaluating on the averaged model
$ python pipeline.py --src_lang bn --tgt_lang en -i inputFolder/ -o outputFolder/ --eval_model  outputFolder/Models/bn2en_step_111000-130000-20.pt --do_eval 
```