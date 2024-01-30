import os
import pprint

import torch
import transformers
from tqdm import tqdm

from math_equivalence import is_equiv
from sample_tokenizer import SampleTokenizer
from util import last_boxed_only_string


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def get_real_sol_idxs(tokens_sol, tokenizer):
    """
    Return the start and stop indexes (inclusive) for everything inside \\boxed{...}
    """
    left_idx, right_idx = None, None
    for i in range(tokens_sol.shape[1]):
        if i < 3:
            continue

        if tokens_sol[0, i].item() and \
                tokens_sol[0, i - 1].item() == 276 and \
                tokens_sol[0, i - 2].item() == 3524:
            # at index i, we have the { of \\boxed{
            left_idx = i + 1  # Don't include the {

        if tokens_sol[0, i].item() == 50256:
            right_idx = i - 2  # don't include the one token before the current one as well (usually the } from \boxed{})

    # Will error if either is not found, which we dont expect
    return left_idx, right_idx


def run_eval(args):
    argsdict = vars(args)
    tokenizer = None
    print(pprint.pformat(argsdict))

    if args.load:
        if args.arch in {'gpt2'}:
            print(f"Loading model from {args.load}")
            model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
            print(f"Successfully loaded model from {args.load}")
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.load)
        elif args.arch in {'bert-base-uncased'}:
            print(f"Loading model from {args.load}")
            model = transformers.BertForMaskedLM.from_pretrained(args.load)
            print(f"Successfully loaded model from {args.load}")
            tokenizer = transformers.BertTokenizer.from_pretrained(args.load)
        elif args.arch in {'t5-base-uncased'}:
            print(f"Loading model from {args.load}")
            model = transformers.T5Model.from_pretrained(args.load)
            print(f"Successfully loaded model from {args.load}")
            tokenizer = transformers.T5Tokenizer.from_pretrained(args.load)
    else:
        if args.arch in {'gpt2'}:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.arch)
        elif args.arch in {'bert-base-uncased'}:
            model = transformers.BertForMaskedLM.from_pretrained(args.arch)
        elif args.arch in {'gpt2'}:
            model = transformers.T5Model.from_pretrained(args.arch)

    eval_data = get_dataset(args)
    for inner_dset in eval_data.datasets:
        inner_dset.tokenizer = tokenizer

    dataloader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )

    model = model.eval()

    loss_moving_average = 0

    outputs = []
    answers = []
    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}

    with torch.no_grad():
        correct = 0
        total = 0
        skipped = 0
        mean_max_probs_correct = []
        mean_max_probs_wrong = []
        for i, batch in enumerate(tqdm(dataloader)):

            if torch.sum(batch['input_ids']) == 0:
                skipped += 1
                print("SKIPPING", batch['fnames'][0])
                continue

            fnames = batch['fnames'][0]
            assert len(fnames) == 1
            fnames_list.append(fnames[0])

            output_ids = model.generate(
                batch['input_ids'],
                num_beams=args.num_beams,
                early_stopping=True,
                temperature=1.0,
                max_length=384 if args.arch == 'gpt2-xl' else 1024
            )

            mean_probs_sol = 0

            output_tokens = get_model_output(batch['input_ids'][0], output_ids[0], tokenizer)

            # Print this iteration
            output_str = tokenizer.decode(output_tokens)
            output_full = output_str
            output_str = last_boxed_only_string(output_str)

            if args.math_mode == "eval_peeking":
                answer_str = last_boxed_only_string(tokenizer.decode(batch['labels'][0]))
            else:
                answer_str = tokenizer.decode(batch['labels'][0])

            output, answer = remove_boxed(output_str), remove_boxed(answer_str)

            print("Problem String:")
            print(tokenizer.decode(batch['input_ids'][0]) + "\n")
            print("Model output:")
            print(output_full)
            print(output)
            print("Correct answer:")
            print(answer)
            print("fname")
            print(fnames)
            print("--------------------------------------------")

            outputs.append(output)
            answers.append(answer)

            equiv = is_equiv(output, answer)
            if equiv:
                correct += 1
                mean_max_probs_correct.append(mean_probs_sol)
            else:
                mean_max_probs_wrong.append(mean_probs_sol)

            total += 1

    subjects = ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra',
                'Precalculus']

    print(f"Average of mean_max_probs_correct = {sum(mean_max_probs_correct)}/{len(mean_max_probs_correct)} = ",
          sum(mean_max_probs_correct) / len(mean_max_probs_correct))
    print(f"Average of mean_max_probs_wrong   = {sum(mean_max_probs_wrong)}/{len(mean_max_probs_wrong)} = ",
          sum(mean_max_probs_wrong) / len(mean_max_probs_wrong))

    # now save outputs and answers
    with open(f"outputs_answers_Temp2e-1_{args.arch}.txt", "w+") as f:
        for k, (output, answer, prob_type, prob_level, fname) in enumerate(
                zip(outputs, answers, fnames_list)):
            f.write("{} OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, output, answer, fname))

        print("#####################")
        f.write("#####################\n")

        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct / total))
        print("Skipped = {}".format(skipped))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct / total))
        f.write("Skipped = {}".format(skipped))

    print()


def get_model_output(context, full_output, tokenizer):
    """
    Given the context and the full model output (context + generated),
    extract just the generated tokens.
    Remove the last token if it is <|endoftext|>
    """
    ret = full_output[len(context):]
    if ret[-1] == tokenizer.eos_token_id:
        ret = ret[:-1]
    return ret


def get_dataset(args):
    eval_datasets = []

    if args.math_dataroot:
        for math_dr in args.math_dataroot:
            flist_find_roots = os.path.join(math_dr, "algebra/flist_testdata_find_roots.txt")
            flist_invert_function = os.path.join(math_dr, "algebra/flist_testdata_invert_function.txt")

            flist_derivatives = os.path.join(math_dr, "calculus/flist_testdata_derivatives.txt")
            flist_integrals = os.path.join(math_dr, "calculus/flist_testdata_integrals.txt")

            flist_polygons = os.path.join(math_dr, "geometry/flist_testdata_polygons.txt")
            flist_triangles = os.path.join(math_dr, "geometry/flist_testdata_triangles.txt")

            flist_determinant = os.path.join(math_dr, "linear_algebra/flist_testdata_determinant.txt")
            flist_orthogonalize_vectors = os.path.join(math_dr,
                                                       "linear_algebra/flist_testdata_orthogonalize_vectors.txt")

            with open(flist_find_roots, "r") as f:
                find_roots_num_files = len(f.readlines())

            with open(flist_invert_function, "r") as f:
                invert_function_num_files = len(f.readlines())

            with open(flist_derivatives, "r") as f:
                derivatives_num_files = len(f.readlines())

            with open(flist_integrals, "r") as f:
                integrals_num_files = len(f.readlines())

            with open(flist_polygons, "r") as f:
                polygons_num_files = len(f.readlines())

            with open(flist_triangles, "r") as f:
                triangles_num_files = len(f.readlines())

            with open(flist_determinant, "r") as f:
                determinant_num_files = len(f.readlines())

            with open(flist_orthogonalize_vectors, "r") as f:
                orthogonalize_vectors_num_files = len(f.readlines())

            if find_roots_num_files:
                eval_datasets.append(SampleTokenizer(
                    math_dataroot=flist_find_roots,
                    tokenizer=None,
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                    mode=args.arch,
                ))

            if invert_function_num_files:
                eval_datasets.append(SampleTokenizer(
                    math_dataroot=flist_invert_function,
                    tokenizer=None,
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                    mode=args.arch,
                ))

            if derivatives_num_files:
                eval_datasets.append(SampleTokenizer(
                    math_dataroot=flist_derivatives,
                    tokenizer=None,
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                    mode=args.arch,
                ))

            if integrals_num_files:
                eval_datasets.append(SampleTokenizer(
                    math_dataroot=flist_integrals,
                    tokenizer=None,
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                    mode=args.arch,
                ))

            if polygons_num_files:
                eval_datasets.append(SampleTokenizer(
                    math_dataroot=flist_polygons,
                    tokenizer=None,
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                    mode=args.arch,
                ))

            if triangles_num_files:
                eval_datasets.append(SampleTokenizer(
                    math_dataroot=flist_triangles,
                    tokenizer=None,
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                    mode=args.arch,
                ))

            if determinant_num_files:
                eval_datasets.append(SampleTokenizer(
                    math_dataroot=flist_determinant,
                    tokenizer=None,
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                    mode=args.arch,
                ))

            if orthogonalize_vectors_num_files:
                eval_datasets.append(SampleTokenizer(
                    math_dataroot=flist_orthogonalize_vectors,
                    tokenizer=None,
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                    mode=args.arch,
                ))

        eval_data = torch.utils.data.ConcatDataset(eval_datasets)
        return eval_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='gpt2', choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST + transformers.BERT_PRETRAINED_MODEL_ARCHIVE_LIST + transformers.T5_PRETRAINED_MODEL_ARCHIVE_LIST)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--num-beams', default=20, type=int)

    # Dataloading
    parser.add_argument('--math_dataroot', default=None, type=str)
    parser.add_argument('--math_mode', default='gpt2-eval', type=str)
    parser.add_argument('--peek-fraction', type=float, default=1.0)

    # Others
    parser.add_argument('--workers', default=4, type=int)

    args = parser.parse_args()

    run_eval(args)
