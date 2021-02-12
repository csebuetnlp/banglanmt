from source.embed import *
from multiprocessing import Pool
import multiprocessing as mp
import time
import random
import shutil
import argparse
import glob
import math

random.seed(3435)


def loadEncoder(cpu=False):
    model_loc = os.path.join(os.environ["LASER"], "models", "bilstm.93langs.2018-12-26.pt")
    print(' - Encoder: loading {}'.format(model_loc))
    global ENCODER

    ENCODER = SentenceEncoder(model_loc,
                              max_sentences=None,
                              max_tokens=12000,
                              sort_kind='quicksort',
                              cpu=cpu)
                              
def encode(ifname, ofname, language):
    with tempfile.TemporaryDirectory() as tmpdir:
        
        tok_fname = os.path.join(tmpdir, 'tok')
        Token(ifname,
                tok_fname,
                lang=language,
                romanize=True if language == 'el' else False,
                lower_case=True, gzip=False,
                verbose=True, over_write=False)
        ifname = tok_fname

        bpe_fname = os.path.join(tmpdir, 'bpe')
        BPEfastApply(ifname,
                    bpe_fname,
                    os.path.join(os.environ["LASER"], "models", "93langs.fcodes"),
                    verbose=True, over_write=False)
        ifname = bpe_fname

        EncodeFile(ENCODER,
                ifname,
                ofname,
                verbose=True, over_write=False,
                buffer_size=10000)

def getLines(filename):
    lines = []
    with open(filename) as f:
        for line in f:
            assert line.strip(), "Empty line found"
            lines.append(line.strip())
    return lines

def writeValidLinePairs(file1, file2):
    f1Lines, f2Lines = [], []

    with open(file1) as f1, open(file2) as f2:
        for line1, line2 in zip(f1, f2):
            if line1.strip() == "" or line2.strip() == "":
                continue

            f1Lines.append(line1.replace('\t', ' ').strip())
            f2Lines.append(line2.replace('\t', ' ').strip())

    
    linePairList = list(dict.fromkeys(zip(f1Lines, f2Lines)))

    with open(file1, 'w') as f1, open(file2, 'w') as f2:
        for linePair in linePairList:
            print(linePair[0].strip(), file=f1)
            print(linePair[1].strip(), file=f2)

def score(prefix, args):
    writeValidLinePairs(f'{prefix}.{args.src_lang}', f'{prefix}.{args.tgt_lang}')

    s = f'''
    python3 \"{os.path.join(os.environ["LASER"], "source", "mine_bitexts.py")}\" \
        \"{prefix}.{args.src_lang}\" \"{prefix}.{args.tgt_lang}\" \
        --src-lang {args.src_lang} --trg-lang {args.tgt_lang} \
        --src-embeddings \"{prefix}.enc.{args.src_lang}\" --trg-embeddings \"{prefix}.enc.{args.tgt_lang}\" \
        --mode score --retrieval max -k 4  \
        --output \"{prefix}.tsv\" \
        --verbose {'--gpu' if not args.cpu else ''}
    '''
    os.system(s)

    os.remove(f'{prefix}.enc.{args.src_lang}')
    os.remove(f'{prefix}.enc.{args.tgt_lang}')

def mergeScores(input_dir, output_file):
    output_lines = []
    for input_file in glob.glob(os.path.join(input_dir, "*tsv")):
        output_lines.extend(getLines(input_file))
    
    _create(output_lines, output_file)

def scoreDir(dirname, out_prefix, args):
    prefixes = [f[:-len(args.tgt_lang) - 1] for f in glob.glob(os.path.join(dirname, f"*{args.tgt_lang}"))]

    for prefix in prefixes:
        encode(f'{prefix}.{args.src_lang}', f'{prefix}.enc.{args.src_lang}', args.src_lang)
        encode(f'{prefix}.{args.tgt_lang}', f'{prefix}.enc.{args.tgt_lang}', args.tgt_lang)
 
    if args.cpu:
        with Pool() as pool:
            pool.starmap(score, [(prefix, args) for prefix in prefixes])
    else:
        for prefix in prefixes:
            score(prefix, args)

    for filename in glob.glob(os.path.join(dirname, "*.enc.*")):
        os.remove(filename)
    
    mergeScores(dirname, os.path.join(os.path.dirname(dirname), out_prefix))
    shutil.rmtree(dirname)

def shufflePairs(srcFile, tgtFile):
    with open(f'{srcFile}.shuffled', 'w') as srcF, open(f'{tgtFile}.shuffled', 'w') as tgtF:
        srcLines, tgtLines = [], []

        with open(srcFile) as f:
            srcLines.extend(f.readlines())

        with open(tgtFile) as f:
            tgtLines.extend(f.readlines())

        assert len(srcLines) == len(tgtLines), "src and tgt line counts dont match"

        indices = list(range(len(srcLines)))
        random.shuffle(indices)

        for i in indices:
            print(srcLines[i].strip(), file=srcF)
            print(tgtLines[i].strip(), file=tgtF)

    shutil.move(f'{srcFile}.shuffled', srcFile)
    shutil.move(f'{tgtFile}.shuffled', tgtFile)

def _create(lines, output_file):
    with open(output_file, 'w') as outf:
        for line in lines:
            print(line.strip(), file=outf)

def createChunks(input_file, output_dir, suffix, chunk_size):
    os.makedirs(output_dir, exist_ok=True)
    input_lines = getLines(input_file)
    no_chunks = math.ceil(len(input_lines) / chunk_size)

    for i in range(no_chunks):
        output_file = os.path.join(output_dir, f"{i}.{suffix}")
        lines = input_lines[i * chunk_size: (i + 1) * chunk_size]
        _create(lines, output_file)
    
def chunkFiles(prefix, dirname, args):
    if os.path.isdir(os.path.join(dirname, "original")):
        shutil.rmtree(os.path.join(dirname, "original"))
       
    shutil.copy(f'{prefix}.{args.src_lang}', f'{prefix}.{args.src_lang}.backup')
    shutil.copy(f'{prefix}.{args.tgt_lang}', f'{prefix}.{args.tgt_lang}.backup')
    
    shufflePairs(f'{prefix}.{args.src_lang}', f'{prefix}.{args.tgt_lang}')

    createChunks(f"{prefix}.{args.src_lang}", os.path.join(dirname, "original"), args.src_lang, args.batch_size)
    createChunks(f"{prefix}.{args.tgt_lang}", os.path.join(dirname, "original"), args.tgt_lang, args.batch_size)

    shutil.move(f'{prefix}.{args.src_lang}.backup', f'{prefix}.{args.src_lang}')
    shutil.move(f'{prefix}.{args.tgt_lang}.backup', f'{prefix}.{args.tgt_lang}')

def batchFilterDir(args):
    for tgtFile in glob.glob(os.path.join(args.input_dir, "**", f"*{args.tgt_lang}"), recursive=True):
        dirname = os.path.dirname(tgtFile)
        prefix = tgtFile[:-len(args.tgt_lang) - 1]
        if not os.path.isfile(f'{prefix}.{args.src_lang}'):
            continue

        chunkFiles(prefix, dirname, args)
        out_prefix = os.path.basename(prefix)
        tsv_name = out_prefix + ".merged.tsv"
        scoreDir(os.path.join(dirname, "original"), tsv_name, args)

        outDir = dirname.replace(os.path.normpath(args.input_dir), os.path.normpath(args.output_dir), 1)
        os.makedirs(outDir, exist_ok=True)
        passed = failed = 0
        
        shutil.move(os.path.join(dirname, tsv_name), os.path.join(outDir, tsv_name))

        with open(os.path.join(outDir, f'{out_prefix}.passed.{args.src_lang}'), 'w') as psrc, \
            open(os.path.join(outDir, f'{out_prefix}.passed.{args.tgt_lang}'), 'w') as ptgt, \
            open(os.path.join(outDir, f'{out_prefix}.failed.{args.src_lang}'), 'w') as fsrc, \
            open(os.path.join(outDir, f'{out_prefix}.failed.{args.tgt_lang}'), 'w') as ftgt:

            with open(os.path.join(outDir, tsv_name)) as f:
                for line in f:
                    score, srcLine, tgtLine = line.split('\t')
                    
                    if float(score) > args.thresh:
                        print(srcLine.strip(), file=psrc)
                        print(tgtLine.strip(), file=ptgt)
                        passed += 1
                    else:
                        print(srcLine.strip(), file=fsrc)
                        print(tgtLine.strip(), file=ftgt)
                        failed += 1
        
        print(f'Passed Sentences: {passed}')
        print(f'Failed Sentences: {failed}')

    

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

    parser.add_argument('--thresh', type=float, default=.95, help='threshold')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')

    parser.add_argument('--cpu', action='store_true',
        help='Run on cpu')

    args = parser.parse_args()
    assert args.input_dir != args.output_dir, "input and output directories cant be the same."
    loadEncoder(args.cpu)
    batchFilterDir(args)
    