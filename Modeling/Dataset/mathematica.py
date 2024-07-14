import os

import torch
from tqdm import tqdm

from base_math_dataset import BaseMathDataset
from util import _clean_numbers, last_boxed_only_string


class Mathematica(BaseMathDataset):
    def __len__(self):
        return int(len(self.samples))

    def initialize(self):
        with open(self.dataroot, 'r') as fp:
            all_filenames = fp.readlines()

        print(f"{self.__class__.__name__}: Loading samples from {len(all_filenames)}")
        samples_raw = []
        for fname in tqdm(all_filenames):
            fname = fname.strip()
            print(fname)

            if not os.path.isfile(fname):
                print(f"Skipping {fname}")
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
                            question = curr_section
                        else:
                            answers.append(curr_section)
                        curr_section = ""
                        reading_question = False
                    else:
                        curr_section += line

                for a in answers:
                    samples_raw.append((question, a, fname))

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

        inputs = torch.cat([
            question_ids,
            sep_ids,
            answer_ids
        ], dim=0)

        label_ids = torch.cat([
            torch.ones_like(question_ids) * -100,
            torch.ones_like(sep_ids) * -100,
            answer_ids.clone()
        ], dim=0)

        if self.mode in {'bert-base-uncased'}:
            random_tensor = torch.rand(inputs['input_ids'].shape)
            masked_tensor = (random_tensor < 0.15) * (inputs['input_ids'] != 101) * (inputs['input_ids'] != 102) * (
                        inputs['input_ids'] != 0)
            nonzero_indices = []
            for i in range(len(masked_tensor)):
                nonzero_indices.append(torch.flatten(masked_tensor[i].nonzero()).tolist())
            for i in range(len(inputs['input_ids'])):
                inputs['input_ids'] = inputs['input_ids'][i, nonzero_indices[i]] != 103

        input_ids = inputs['input_ids'].tolist()
        label_ids = label_ids.tolist()

        return {
            'input_ids_list': input_ids,
            'label_ids_list': label_ids,
            'attention_mask_list': inputs['attention_mask'] if self.mode in {'bert-base-uncased', 't5-base-uncased'} else None
        }

    def clean_filter_sample_eval(self, sample):
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

        inputs = torch.cat([
            question_ids,
            sep_ids
        ], dim=0)

        label_ids = torch.cat([answer_final_ids.clone()], dim=0)

        if inputs.shape[0] + label_ids.shape[0] >  self.max_tokens:
            return None

        if self.mode in {'bert-base-uncased'}:
            random_tensor = torch.rand(inputs['input_ids'].shape)
            masked_tensor = (random_tensor < 0.15) * (inputs['input_ids'] != 101) * (inputs['input_ids'] != 102) * (
                    inputs['input_ids'] != 0)
            nonzero_indices = []
            for i in range(len(masked_tensor)):
                nonzero_indices.append(torch.flatten(masked_tensor[i].nonzero()).tolist())
            for i in range(len(inputs['input_ids'])):
                inputs['input_ids'] = inputs['input_ids'][i, nonzero_indices[i]] != 103

        return {
            'input_ids_list': inputs['input_ids'].tolist(),
            'label_ids_list': label_ids.tolist(),
            'attention_mask_list': inputs['attention_mask'] if self.mode in {'bert-base-uncased', 't5-base-uncased'} else None
        }
