import numpy as np
import argparse
import sys
import shutil
import pyonmttok
import os
import glob
import math
from tqdm import tqdm

def createFolders(args):
    required_dirnames = [
        "data",
        "Outputs",
        "temp",
        "Preprocessed",
        "Reports",
        "Models"
    ]

    # do cleanup first
    for dirname in required_dirnames[:3]:
        if os.path.isdir(os.path.join(args.output_dir, dirname)):
            shutil.rmtree(os.path.join(args.output_dir, dirname))

    
    if os.path.isdir(os.path.join(args.output_dir, "Preprocessed")):
        shutil.rmtree(os.path.join(args.output_dir, "Preprocessed"))

    for dirname in required_dirnames:
        os.makedirs(os.path.join(args.output_dir, dirname), exist_ok=True)

def _merge(args, data_type):
    with open(os.path.join(args.output_dir, "data", f"src-{data_type}.txt"), 'w') as srcF, \
        open(os.path.join(args.output_dir, "data", f"tgt-{data_type}.txt"), 'w') as tgtF:
        
        for src_file in glob.glob(os.path.join(args.input_dir, "data", f"*.{data_type}.{args.src_lang}")):
            tgt_file_prefix = src_file.rsplit(f".{data_type}.{args.src_lang}", 1)[0] + f".{data_type}.{args.tgt_lang}"
            tgt_files = glob.glob(tgt_file_prefix + "*")

            if tgt_files:
                # when multiple references are present, pick the first one
                tgt_file = tgt_files[0]
                
                with open(src_file) as f:
                    for line in f:
                        print(line.strip(), file=srcF)
                
                with open(tgt_file) as f:
                    for line in f:
                        print(line.strip(), file=tgtF)

def _move(args, dataset_category):
    for src_file in glob.glob(os.path.join(args.input_dir, "data", f"*.{dataset_category}.{args.src_lang}")):
        tgt_file_prefix = src_file.rsplit(f".{dataset_category}.{args.src_lang}", 1)[0] + f".{dataset_category}.{args.tgt_lang}"
        tgt_files = glob.glob(tgt_file_prefix + "*")
        if tgt_files:
            shutil.copy(
                src_file,
                os.path.join(
                    args.output_dir,
                    "Outputs",
                    f".src-{dataset_category}.txt".join(
                        os.path.basename(src_file).rsplit(f".{dataset_category}.{args.src_lang}", 1)
                    )
                ) 
            )
            for tgt_file in tgt_files:
                shutil.copy(
                tgt_file, 
                os.path.join(
                    args.output_dir,
                    "Outputs",
                    f".tgt-{dataset_category}.txt".join(
                        os.path.basename(tgt_file).rsplit(f".{dataset_category}.{args.tgt_lang}", 1)
                    )
                )
            )
    
def moveRawData(args):
    # move vocab models
    shutil.copy(
        os.path.join(args.input_dir, "vocab", f"{args.src_lang}.model"),
        os.path.join(args.output_dir, "Preprocessed", "srcSPM.model")
    )
    shutil.copy(
        os.path.join(args.input_dir, "vocab", f"{args.tgt_lang}.model"),
        os.path.join(args.output_dir, "Preprocessed", "tgtSPM.model")
    )

    vocab_cmd = [
        "spm_export_vocab --model",
        os.path.join(args.output_dir, "Preprocessed", "srcSPM.model"),
        "| tail -n +4 >",
        os.path.join(args.output_dir, "Preprocessed", "srcSPM.vocab")
    ]
    os.system(" ".join(vocab_cmd))

    vocab_cmd = [
        "spm_export_vocab --model",
        os.path.join(args.output_dir, "Preprocessed", "tgtSPM.model"),
        "| tail -n +4 >",
        os.path.join(args.output_dir, "Preprocessed", "tgtSPM.vocab")
    ]
    os.system(" ".join(vocab_cmd))

    if args.do_train:
        _merge(args, "train")
        _merge(args, "valid")
        
        if not glob.glob(os.path.join(args.input_dir, "data", f"*.valid.{args.src_lang}")):       
            np.random.seed(3435)
            sampledCount = 0
            
            with open(os.path.join(args.output_dir, "data", "src-train.txt.backup"), 'w') as srcT, \
                open(os.path.join(args.output_dir, "data", "tgt-train.txt.backup"), 'w') as tgtT, \
                open(os.path.join(args.output_dir, "data", "src-valid.txt"), 'w') as srcV, \
                open(os.path.join(args.output_dir, "data", "tgt-valid.txt"), 'w') as tgtV, \
                open(os.path.join(args.output_dir, "data", "src-train.txt")) as srcO, \
                open(os.path.join(args.output_dir, "data", "tgt-train.txt")) as tgtO:
                
                for srcLine, tgtLine in zip(srcO, tgtO):
                    if sampledCount < args.validation_samples:
                        if np.random.random() > .5:
                            print(srcLine.strip(), file=srcV)
                            print(tgtLine.strip(), file=tgtV)
                            sampledCount += 1
                            continue
                    
                    print(srcLine.strip(), file=srcT)
                    print(tgtLine.strip(), file=tgtT)

            shutil.move(
                os.path.join(args.output_dir, "data", "src-train.txt.backup"),
                os.path.join(args.output_dir, "data", "src-train.txt")
            )
            shutil.move(
                os.path.join(args.output_dir, "data", "tgt-train.txt.backup"),
                os.path.join(args.output_dir, "data", "tgt-train.txt")
            )


    if args.do_eval:
        _move(args, "valid")
        _move(args, "test")
        
def _lc(input_file):
    lc = 0
    with open(input_file) as f:
        for _ in f:
            lc += 1
    return lc
    

def spmOperate(args, fileType, tokenize):
    if tokenize:
        modelName = os.path.join(args.output_dir, "Preprocessed", f"{fileType}SPM.model")
        input_files = glob.glob(os.path.join(args.output_dir, "Outputs", f'*{fileType}-*'))

        for input_file in input_files:
            spm_cmd = [
                f"spm_encode --model=\"{modelName}\"",
                f"--output_format=piece",
                f"< \"{input_file}\" > \"{input_file}.tok\""
            ]
            os.system(" ".join(spm_cmd))
            os.remove(input_file)

    else:
        modelName = os.path.join(args.output_dir, "Preprocessed", f"tgtSPM.model")
        for input_file in glob.glob(os.path.join(args.output_dir, "Outputs", f'*{fileType}-*.tok')):
            spm_cmd = [
                f"spm_decode --model=\"{modelName}\"",
                f"< \"{input_file}\" > \"{'.detok'.join(input_file.rsplit('.tok', 1))}\""
            ]
            os.system(" ".join(spm_cmd))
            os.remove(input_file)
            post_cmd = f"""sed 's/▁/ /g;s/  */ /g' -i \"{'.detok'.join(input_file.rsplit('.tok', 1))}\""""
            os.system(post_cmd)
        
        
def tokenize(args):
    spmOperate(args, 'src', tokenize=True)
    spmOperate(args, 'tgt', tokenize=True)
            
def detokenize(args):        
    spmOperate(args, 'tgt', tokenize=False)
    spmOperate(args, 'pred', tokenize=False)

def processData(args, tokenization):
    if tokenization:
        createFolders(args)
        moveRawData(args)
        tokenize(args)
    else:
        detokenize(args)

