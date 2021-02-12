import os
import re
import subprocess
import traceback
import time
import shutil
import argparse
import glob
from dataProcessor import processData

OPENNMTDIR = os.path.join(os.path.dirname(__file__), "..")

def preprocess(args):
    cmd = f'''
        python \"{os.path.join(OPENNMTDIR, "preprocess.py")}\" \
            -train_src \"{os.path.join(args.output_dir, "data", "src-train.txt.tok")}\" \
            -train_tgt \"{os.path.join(args.output_dir, "data", "tgt-train.txt.tok")}\" \
            -valid_src \"{os.path.join(args.output_dir, "data", "src-valid.txt.tok")}\" \
            -valid_tgt \"{os.path.join(args.output_dir, "data", "tgt-valid.txt.tok")}\" \
            -save_data \"{os.path.join(args.output_dir, "Preprocessed", "processed")}\" \
            -src_vocab \"{os.path.join(args.output_dir, "Preprocessed", "srcSPM.vocab")}\" \
            -tgt_vocab \"{os.path.join(args.output_dir, "Preprocessed", "tgtSPM.vocab")}\" \
            -src_seq_length {args.src_seq_length} \
            -tgt_seq_length {args.tgt_seq_length} -filter_valid
    '''
    os.system(cmd)

def train(args):
    cmd = f'''
        python \"{os.path.join(OPENNMTDIR, "train.py")}\" \
            -data \"{os.path.join(args.output_dir, "Preprocessed", "processed")}\" \
            -save_model \"{os.path.join(args.output_dir, "Models", args.model_prefix)}\" \
            -layers {args.layers} -rnn_size {args.rnn_size} -word_vec_size {args.word_vec_size} -transformer_ff {args.transformer_ff} -heads {args.heads}  \
			-encoder_type transformer -decoder_type transformer -position_encoding \
            -train_steps {args.train_steps} -max_generator_batches 2 -dropout 0.1 \
            -batch_size {args.train_batch_size} -batch_type tokens -normalization tokens -accum_count {args.gradient_accum} \
            -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps {args.warmup_steps} -learning_rate {args.learning_rate} \
            -max_grad_norm 0 -param_init 0  -param_init_glorot \
			-share_decoder_embeddings  \
            -label_smoothing 0.1 -valid_steps {args.valid_steps} -save_checkpoint_steps {args.save_checkpoint_steps} \
            -world_size {args.world_size} -gpu_ranks {" ".join(args.gpu_ranks)} {"-train_from " + args.train_from if args.train_from else ""}  
    '''
    os.system(cmd)

def average_models(args):
    step_count = lambda p: int(re.search(r"_step_(\d+)", p).group(1))
    model_paths = sorted(
            glob.glob(os.path.join(args.output_dir, "Models", f"{args.model_prefix}*.pt")),
            key=step_count
    )
    if len(model_paths) > args.average_last:
        model_paths = model_paths[-args.average_last:]
        output_path = (
            model_paths[0].rsplit("_step_")[0] + 
            f"_step_{step_count(model_paths[0])}-{step_count(model_paths[-1])}-{args.average_last}.pt"
        )
        cmd = [
            f"python \"{os.path.join(OPENNMTDIR, 'tools', 'average_models.py')}\"",
            f"-models {' '.join(model_paths)}",
            f"-output {output_path}"
        ]
        os.system(" ".join(cmd))

def _translate(args, modelName, inputFile, outputFile):
    cmd = f'''
        python \"{os.path.join(OPENNMTDIR, "translate.py")}\" \
            -model {modelName} \
            -src \"{inputFile}\" \
            -output \"{outputFile}\" \
            -replace_unk -verbose -max_length {args.tgt_seq_length} -batch_size {args.eval_batch_size} -report_bleu -fp32 -gpu 0
    '''
    os.system(cmd)

def translate(model_path, dataset_category, args):
    src_lines, src_map = [], {}
    for src_file in glob.glob(os.path.join(args.output_dir, "Outputs", f'*src-{dataset_category}.txt.tok')):
        with open(src_file) as f:
            lines = f.readlines()
            src_map[src_file] = len(lines)
            src_lines.extend(lines)

    merged_src_file = os.path.join(args.output_dir, "temp", "merged.src")
    merged_tgt_file = os.path.join(args.output_dir, "temp", "merged.tgt")

    with open(merged_src_file, 'w') as f:
        for line in src_lines:
            print(line.strip(), file=f)
    
    _translate(args, model_path, merged_src_file, merged_tgt_file)

    with open(merged_tgt_file) as inpf:
        idx = 0
        lines = inpf.readlines()

        for src_file in src_map:
            pred_file = f"pred-{dataset_category}.txt.tok".join(
                src_file.rsplit(
                    f"src-{dataset_category}.txt.tok", 1
                )
            )

            with open(pred_file, 'w') as outf:
                for _ in range(src_map[src_file]):
                    print(lines[idx].strip(), file=outf)
                    idx += 1

    os.remove(merged_src_file)
    os.remove(merged_tgt_file)

def calculate_scores(args, dataset_category):
    scores = []
    for pred_file in glob.glob(os.path.join(args.output_dir, "Outputs", f'*pred-{dataset_category}.txt.detok')):
        dataset_name = os.path.basename(pred_file).rsplit(
            f".pred-{dataset_category}.txt.detok", 1
        )[0]

        tgt_file_prefix = f".tgt-{dataset_category}.txt.*detok".join(
            pred_file.rsplit(
                f".pred-{dataset_category}.txt.detok", 1
            )
        )
        tgt_files = glob.glob(tgt_file_prefix)
        if tgt_files:
            bleu_cmd = [
                f"perl \"{os.path.join(OPENNMTDIR, 'tools', 'multi-bleu-detok.perl')}\"",
                f"-lc {' '.join(tgt_files)} < {pred_file}"
            ]
            sacre_cmd = [
                f"cat {pred_file}",
                "|",
                f"sacrebleu {' '.join(tgt_files)}"
            ]
            
            try:
                bleu_output = str(subprocess.check_output(" ".join(bleu_cmd), shell=True)).strip()
                bleu_score = bleu_output.splitlines()[-1].split(",")[0].split("=")[1]
            except:
                bleu_score = -1

            try:
                sacre_output = str(subprocess.check_output(" ".join(sacre_cmd), shell=True)).strip()
                sacre_score = sacre_output.splitlines()[-1].split("=")[1].split()[0]
            except:
                sacre_score = -1

            scores.append(
                {
                    "dataset": dataset_name,
                    "bleu": bleu_score,
                    "sacrebleu": sacre_score
                }
            )

    return scores

def write_scores(scores, output_path):
    with open(output_path, 'w') as f:
        for model_name in scores:
            print(model_name, ":", file=f)
            for dataset_score in scores[model_name]:
                print(
                    "",
                    f"Dataset: {dataset_score['dataset']},",
                    f"BLEU: {dataset_score['bleu']},",
                    f"SACREBLEU: {dataset_score['sacrebleu']},",
                    sep="\t",
                    file=f
                )

def evaluate(args):
    if args.model_prefix:
        model_paths = sorted(
            glob.glob(os.path.join(args.output_dir, "Models", f"{args.model_prefix}*.pt")),
            key=lambda p: int(re.search(r"_step_(\d+)", p).group(1))
        )
        model_scores = {} 
        for model_path in model_paths:
            translate(model_path, "valid", args)
            processData(args, False)
            scores = calculate_scores(args, "valid")
            model_scores[os.path.basename(model_path)] = scores
        
        write_scores(
            model_scores,
            os.path.join(
                args.output_dir, 
                "Reports", f"{args.model_prefix}.valid.{args.src_lang}2{args.tgt_lang}.log"
            )
        )
    
    if args.eval_model:
        model_scores = {}
        translate(args.eval_model, "test", args)
        processData(args, False)
        scores = calculate_scores(args, "test")
        model_scores[os.path.basename(args.eval_model)] = scores

        write_scores(
            model_scores,
            os.path.join(
                args.output_dir, "Reports", f"{os.path.basename(args.eval_model)}.test.{args.src_lang}2{args.tgt_lang}.log"
            )
        )


def main(args):
    processData(args, True)
    if args.do_preprocess:
        preprocess(args)
    if args.do_train:
        train(args)
    if args.model_prefix and args.average_last:
        average_models(args)
    if args.do_eval:
        evaluate(args)
    
                            
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
    
    parser.add_argument(
        '--validation_samples', type=int, default=5000, 
        help='no. of validation samples to take out from train dataset when no validation data is present')
    
    parser.add_argument(
        '--src_seq_length', type=int, default=200, 
        help='maximum source sequence length')

    parser.add_argument(
        '--tgt_seq_length', type=int, default=200, 
        help='maximum target sequence length')

    parser.add_argument(
        '--model_prefix', type=str, 
        help='Prefix of the model to save')

    parser.add_argument(
        '--eval_model', type=str, metavar="PATH", 
        help='Path to the specific model to evaluate')

    parser.add_argument(
        '--train_steps', type=int, default=120000, 
        help='no of training steps')

    parser.add_argument(
        '--train_batch_size', type=int, default=12288, 
        help='training batch size (in tokens)')

    parser.add_argument(
        '--eval_batch_size', type=int, default=8, 
        help='evaluation batch size (in sentences)')

    parser.add_argument(
        '--gradient_accum', type=int, default=2, 
        help='gradient accum')

    parser.add_argument(
        '--warmup_steps', type=int, default=4000, 
        help='warmup steps')

    parser.add_argument(
        '--learning_rate', type=int, default=2, 
        help='learning rate')

    parser.add_argument(
        '--layers', type=int, default=6, 
        help='layers')

    parser.add_argument(
        '--rnn_size', type=int, default=512, 
        help='rnn size')

    parser.add_argument(
        '--word_vec_size', type=int, default=512, 
        help='word vector size')

    parser.add_argument(
        '--transformer_ff', type=int, default=2048, 
        help='transformer feed forward size')

    parser.add_argument(
        '--heads', type=int, default=8, 
        help='no of heads')

    parser.add_argument(
        '--valid_steps', type=int, default=2000, 
        help='validation interval')

    parser.add_argument(
        '--save_checkpoint_steps', type=int, default=1000, 
        help='model saving interval')

    parser.add_argument(
        '--average_last', type=int, default=20, 
        help='average last X models')

    parser.add_argument(
        '--world_size', type=int, default=4, 
        help='world size')

    parser.add_argument(
        '--gpu_ranks', type=int, nargs="*", default=[0, 1, 2, 3], 
        help='gpu ranks')

    parser.add_argument(
        '--train_from', type=str, default="", 
        help='start training from this checkpoint')
    
    parser.add_argument('--do_train', action='store_true',
        help='Run training')
    parser.add_argument('--do_eval', action='store_true',
        help='Run evaluation')
    parser.add_argument('--do_preprocess', action='store_true',
        help='Run the preprocessor to create binary files. Delete previous files if theres any.')
    
    parser.add_argument(
        '--nbest', type=int, default=32, 
        help='sentencepiece nbest size')
    parser.add_argument(
        '--alpha', type=float, default=0.1, 
        help='sentencepiece alpha')
    
    args = parser.parse_args()
    main(args)
    

    
