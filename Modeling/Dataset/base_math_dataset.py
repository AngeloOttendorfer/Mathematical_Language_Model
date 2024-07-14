import os
import random
import time

import torch
import torch.nn.functional as F


class MathDataset(torch.utils.data.Dataset):
    def __init__(self, math_dataroot, tokenizer, mode, max_tokens, sample_inputs=None, encodings=None, packing=None,
                 randomize=None, pack_end=None, clean_numbers=True, peek_fraction=(0.1, 1.0)):
        self.math_dataroot = math_dataroot
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_tokens = max_tokens
        self.sample_inputs = sample_inputs
        self.encodings = encodings
        self.packing = packing
        self.randomize = randomize
        self.pack_end = pack_end
        self.clean_numbers = clean_numbers
        self.peek_fraction = peek_fraction

        if self.mode in {'gpt2', 'bert-base-uncased', 't5-base-uncased'}:
            self.clean_sample = self.clean_filter_sample
            self.packing = True
            self.randomize = True
            self.include_fnames = False
            self.pack_end = True
        elif self.mode in {'gpt2-eval', 'bert-base-uncased-eval', 't5-base-uncased'}:
            self.clean_sample = self.clean_filter_sample_eval
            self.packing = True
            self.randomize = False
            self.include_fnames = True
            self.pack_end = True

        self.bad_fnames = set()
        self.i = 0

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        # Each worker needs a different seed
        random.seed(os.getpid() + time.time() + random.random())
        if self.model in {'gpt2'}:
            # Sampling with replacement.
            # We need to pack random elements to get close to self.max_tokens
            curr_input_ids = []
            curr_label_ids = []
            curr_fnames = []
            num_samples = 0
            while len(curr_input_ids) + 1 <= self.max_tokens and len(curr_label_ids) + 1 <= self.max_tokens:
                curr_sample, fname = self.get_random_sample()
                if curr_sample is None:
                    # Only in evaluation modes
                    return {
                        "input_ids": torch.zeros([self.max_tokens]),
                        "labels": torch.zeros([self.max_tokens]),
                        "fnames": [fname]
                    }
                if not self.pack_end and (
                        (len(curr_input_ids) + 1 + len(curr_sample['input_ids_list']) > self.max_tokens) or
                        (len(curr_label_ids) + 1 + len(curr_sample['label_ids_list']) > self.max_tokens)):
                    # Do not include curr_sample if either the input_ids or the label_ids will run off the end
                    break

                # Add curr_sample to the current inputs and labels
                curr_input_ids.extend(curr_sample['input_ids_list'])
                curr_label_ids.extend(curr_sample['label_ids_list'])
                curr_fnames.append(fname)

                num_samples += 1

                # Break on the first iteration if we don't want to do packing
                if not self.packing:
                    break

                input_ids = torch.LongTensor(curr_input_ids)
                label_ids = torch.LongTensor(curr_label_ids)
                input_ids = input_ids[:self.max_tokens]
                label_ids = label_ids[:self.max_tokens]

                # Padding
                if len(curr_input_ids) < self.max_tokens and 'eval' not in self.mode:
                    num_to_pad = self.max_tokens - len(curr_input_ids)
                    input_ids = F.pad(input_ids, [0, num_to_pad], mode='constant', value=self.tokenizer.pad_token_id)
                if len(curr_label_ids) < self.max_tokens and 'eval' not in self.mode:
                    num_to_pad = self.max_tokens - len(curr_label_ids)
                    label_ids = F.pad(label_ids, [0, num_to_pad], mode='constant', value=-100)

                if self.include_fnames:
                    return {
                        "input_ids": input_ids,
                        "labels": label_ids,
                        "fnames": curr_fnames
                    }
                else:
                    return {
                        "input_ids": input_ids,
                        "labels": label_ids
                    }

        # In comparison to gpt2 the attention mask and in case of bert the token type ids are additionally included
        elif self.mode in {'bert-base-uncased', 't5-base-uncased'}:
            input_ids = self.encodings['input_ids'][index]
            labels = self.encodings['labels'][index]
            attention_mask = self.encodings['attention_mask'][index]
            token_type_ids = self.encodings['token_type_ids'][index]
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids if self.mode in {'bert-base-uncased'} else None
            }


    def get_random_sample(self):
        """
        :return: a random sample (only used for training)
        """
        random_sample = None
        while random_sample is None:
            if self.randomize:
                q, a, fname = random.choice(self.samples)
            else:
                q, a, fname = self.samples[self.i]
                self.i = (self.i + 1) % len(self.samples)

            random_sample = self.clean_sample(q, a)

            if not self.randomize:
                break

        return random_sample, fname
