import numpy as np
import argparse
import sys
import shutil
import pyonmttok
import os
import glob
import math
from tqdm import tqdm

def get_linepairs(args, data_type):
    linepairs = set()
    
    for src_file in glob.glob(os.path.join(args.input_dir, "data", f"*.{data_type}.{args.src_lang}")):
        tgt_file_prefix = src_file.rsplit(f".{data_type}.{args.src_lang}", 1)[0] + f".{data_type}.{args.tgt_lang}"
        tgt_files = glob.glob(tgt_file_prefix + "*")

        if tgt_files:
            for tgt_file in tgt_files:
                with open(src_file) as fs, open(tgt_file) as ft:
                    for src_line, tgt_line in zip(fs, ft):
                        linepairs.add(
                            (src_line.strip(), tgt_line.strip())
                        )
    return linepairs
        
def main(args):
    exclude_linepairs = set()
    exclude_linepairs.update(
        get_linepairs(args, "valid")
    )
    exclude_linepairs.update(
        get_linepairs(args, "test")
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"corpus.train.{args.src_lang}"), 'w') as srcF, \
        open(os.path.join(args.output_dir,  f"corpus.train.{args.tgt_lang}"), 'w') as tgtF:
        
        for src_file in glob.glob(os.path.join(args.input_dir, "data", f"*.train.{args.src_lang}")):
            tgt_file_prefix = src_file.rsplit(f".train.{args.src_lang}", 1)[0] + f".train.{args.tgt_lang}"
            tgt_files = glob.glob(tgt_file_prefix + "*")

            if tgt_files:
                # when multiple references are present, pick the first one
                tgt_file = tgt_files[0]
                
                with open(src_file) as fs, open(tgt_file) as ft:
                    for src_line, tgt_line in zip(fs, ft):
                        src_line = src_line.strip()
                        tgt_line = tgt_line.strip()

                        if (src_line, tgt_line) not in exclude_linepairs:
                            print(src_line, file=srcF)
                            print(tgt_line, file=tgtF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', '-i', type=str,
        required=True,
        metavar='PATH',
        help="Input directory")

    parser.add_argument(
        '--output_dir', '-o', type=str,
        required=True,
        metavar='PATH',
        help="Output directory")

    parser.add_argument(
        '--src_lang', type=str,
        required=True,
        help="Source language")

    parser.add_argument(
        '--tgt_lang', type=str,
        required=True,
        help="Target language")

    args = parser.parse_args()
    main(args)
        


