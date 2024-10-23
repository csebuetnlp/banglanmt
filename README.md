# Bangla-NMT

This repository contains the code and data of the paper titled [**"Not Low-Resource Anymore: Aligner Ensembling, Batch Filtering, and New Datasets for Bengali-English Machine Translation"**](https://www.aclweb.org/anthology/2020.emnlp-main.207/) published in *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), November 16 - November 20, 2020.*

## Updates

* The base translation models are now available for download.
* The training code has been refactored to support [OpenNMT-py 2.2.0](https://github.com/OpenNMT/OpenNMT-py).
* [Colab Notebook](https://colab.research.google.com/drive/1TPkYXEWrf_dUjq-1qpapkreLc7JOug9E?usp=sharing) added for the inference module.

## Table of Contents

- [Bangla-NMT](#bangla-nmt)
  - [Updates](#updates)
  - [Table of Contents](#table-of-contents)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Dependencies](#dependencies)
  - [Segmentation](#segmentation)
  - [Batch-filtering](#batch-filtering)
  - [Training & Evaluation](#training--evaluation)
  - [License](#license)
  - [Citation](#citation)


## Datasets
  Download the dataset from [here](https://docs.google.com/uc?export=download&id=1FLlC0NNXFKVGaVM3-cYW-XEx8p8eV3Wm). This includes:
* Our original 2.75M training corpus (`2.75M/`)
* [Preprocessed](training/preprocessing) training corpus (`data/`)
* RisingNews dev/test sets (`data/`)
* Preprocessed sipc dev/test sets (`data/`)
* Sentencepiece vocabulary models for Bengali and English (`vocab/`) 

## Models

The base-sized transformer model (6 layers, 8 attention heads) checkpoints can be found below: 

* [Bengali to English](https://docs.google.com/uc?export=download&id=1nYKua6_q7W-WK-Xwng_DjoLoZ0k1HgjB)
* [English to Bengali](https://docs.google.com/uc?export=download&id=1uX8nL3yeosmK3YVCRHNJolv861-fCCbi)
* [Sentencepiece vocabulary files](vocab.tar.bz2)

To evaluate these models on new datasets, please refer to [here](https://github.com/csebuetnlp/banglanmt/tree/master/training). You can also use the [Colab Notebook](https://colab.research.google.com/drive/1TPkYXEWrf_dUjq-1qpapkreLc7JOug9E?usp=sharing) for direct inference.

## Dependencies
* Python 3.7.3
* [PyTorch 1.2](http://pytorch.org/)
* [Cython](https://pypi.org/project/Cython/)
* [Faiss](https://github.com/facebookresearch/faiss)
* [FastBPE](https://github.com/glample/fastBPE)
* [sentencepiece](https://github.com/google/sentencepiece) (`Install CLI`)
* [transliterate](https://pypi.org/project/transliterate) 
* [regex](https://pypi.org/project/regex/)
* [torchtext](https://pypi.org/project/torchtext) (`pip install torchtext==0.4.0`)
* [sacrebleu](https://pypi.org/project/sacrebleu)
* [aksharamukha](https://pypi.org/project/aksharamukha)


## Segmentation
  * See [segmentation module.](segmentation/)

## Batch-filtering
  * See [batch-filtering module.](batch_filtering/)

## Training & Evaluation
  * See [training and evaluation module.](training/)
  * Try out the models in [Google Colaboratory.](https://colab.research.google.com/drive/1TPkYXEWrf_dUjq-1qpapkreLc7JOug9E?usp=sharing)

## License
Contents of this repository are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

## Citation
If you use this dataset or code modules, please cite the following paper:
```
@inproceedings{hasan-etal-2020-low,
    title = "Not Low-Resource Anymore: Aligner Ensembling, Batch Filtering, and New Datasets for {B}engali-{E}nglish Machine Translation",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Samin, Kazi  and
      Hasan, Masum  and
      Basak, Madhusudan  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.207",
    doi = "10.18653/v1/2020.emnlp-main.207",
    pages = "2612--2623",
    abstract = "Despite being the seventh most widely spoken language in the world, Bengali has received much less attention in machine translation literature due to being low in resources. Most publicly available parallel corpora for Bengali are not large enough; and have rather poor quality, mostly because of incorrect sentence alignments resulting from erroneous sentence segmentation, and also because of a high volume of noise present in them. In this work, we build a customized sentence segmenter for Bengali and propose two novel methods for parallel corpus creation on low-resource setups: aligner ensembling and batch filtering. With the segmenter and the two methods combined, we compile a high-quality Bengali-English parallel corpus comprising of 2.75 million sentence pairs, more than 2 million of which were not available before. Training on neural models, we achieve an improvement of more than 9 BLEU score over previous approaches to Bengali-English machine translation. We also evaluate on a new test set of 1000 pairs made with extensive quality control. We release the segmenter, parallel corpus, and the evaluation set, thus elevating Bengali from its low-resource status. To the best of our knowledge, this is the first ever large scale study on Bengali-English machine translation. We believe our study will pave the way for future research on Bengali-English machine translation as well as other low-resource languages. Our data and code are available at https://github.com/csebuetnlp/banglanmt.",
}
```
