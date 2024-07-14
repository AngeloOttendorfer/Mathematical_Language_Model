import os

import torch
from tqdm import tqdm

from Dataset.base_math_dataset import BaseMathDataset
from Dataset.util import _clean_numbers, last_boxed_only_string


class Mathematica(BaseMathDataset):

    def __len__(self):
        return int(len(self.samples) * self.len_multiplier)

    def initialize(self):
        """
        Set up self.samples by loading from the dataroot
        """

        with open(self.dataroot, 'r') as fp:
            all_filenames = fp.readlines()

        print(f"{self.__class__.__name__}: Loading samples from {len(all_filenames)} files.")
        samples_raw = []
        for fname in tqdm(all_filenames):
            fname = fname.rstrip()
            print("fname: " + str(fname))
            if not os.path.isfile(fname):
                print(f"SKIPPING {fname}")
                continue
            with open(fname, 'r') as fp:
                question = ""
                answers = []
                reading_question = True
                curr_section = ""
                for line in fp:
                    if line == "Problem:\n":
                        reading_question = True
                    elif line == "Answer:\n":
                        if reading_question:
                            # curr_section contains Q
                            question = curr_section
                        else:
                            # curr_section contains an A
                            answers.append(curr_section)
                        curr_section = ""
                        reading_question = False
                    else:
                        curr_section += line

                # The last answer needs to be recorded.
                answers.append(curr_section)
                fname = os.path.basename(fname)

            for a in answers:
                samples_raw.append((question, a, fname))

        # manager = Manager()
        # samples_raw = manager.list(samples_raw)
        self.samples = samples_raw
        del samples_raw

        print(f"{self.__class__.__name__}: Loaded {len(self.samples)} samples.")

    def clean_filter_sample(self, sample):
        if sample == None:
            return None

        question, answer = sample

        if self.clean_numbers:
            question = _clean_numbers(question)
            answer = _clean_numbers(answer)

        answer_final = last_boxed_only_string(answer)

        question_ids = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))
        sep_ids = torch.LongTensor(self.tokenizer.encode("\nFULL SOLUTION.\n", verbose=False))
        answer_ids = self.tokenizer.encode(answer, verbose=False)
        answer_ids.append(self.tokenizer.eos_token_id)

        if self.mode in {'t5-base-uncased'}:
            answer_ids = answer_ids['input_ids']
            answer_ids[answer_ids == self.tokenizer.pad_token_id] = -100

        answer_ids = torch.LongTensor(answer_ids)

        input_ids = torch.cat([
            question_ids,
            sep_ids,
            answer_ids
        ], dim=0)

        label_ids = torch.cat([
            torch.ones_like(question_ids) * -100,
            torch.ones_like(sep_ids) * -100,
            answer_ids.clone()
        ], dim=0)

        print("input_ids: " + str(input_ids))
        print("label_ids: " + str(label_ids))

        return {
            'input_ids_list': input_ids,
            'label_ids_list': label_ids,
        }

    def clean_filter_sample_eval(self, sample):
        print("entering clean_filter_sample_eval")
        if sample is None:
            return None

        question, answer = sample
        if self.clean_numbers:
            question = _clean_numbers(question)
            answer = _clean_numbers(answer)

        assert not answer.isspace()

        question_ids = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))
        sep_ids = torch.LongTensor(self.tokenizer.encode("\nFULL SOLUTION.\n", verbose=False))
        answer_final_ids = torch.LongTensor(self.tokenizer.encode(answer, verbose=False))

        input_ids = torch.cat([
            question_ids,
            sep_ids
        ], dim=0)

        label_ids = torch.cat([answer_final_ids.clone()], dim=0)

        if input_ids.shape[0] + label_ids.shape[0] > self.max_tokens:
            return None

        return {
            'input_ids_list': input_ids,
            'label_ids_list': label_ids,
        }
