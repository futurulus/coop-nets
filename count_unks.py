#!/usr/bin/env python
from stanza.research import config, evaluate, metrics, output, instance

import color_instances
from tokenizers import TOKENIZERS
from vectorizers import SequenceVectorizer

parser = config.get_options_parser()
parser.add_argument('--tokenizer', choices=TOKENIZERS.keys(), default='unigram',
                    help='The tokenization/preprocessing method to use before unk replacement.')
parser.add_argument('--unk_threshold', type=int, default=1,
                    help="The maximum number of occurrences of a token in the training data "
                         "before it's assigned a non-<unk> token index. 0 means nothing in "
                         "the training data is to be treated as unknown words; 1 means "
                         "single-occurrence words (hapax legomena) will be replaced with <unk>.")
parser.add_argument('--data_source', default='filtered_dev',
                    choices=color_instances.SOURCES.keys(),
                    help='The type of data to use.')


def count_unks():
    options = config.options()

    print('Data source: {}'.format(options.data_source))
    print('Unk threshold: {}'.format(options.unk_threshold))
    print('Tokenizer: {}'.format(options.tokenizer))

    print('')
    print('Loading data')
    train_insts = color_instances.SOURCES[options.data_source].train_data(listener=True)
    eval_insts = color_instances.SOURCES[options.data_source].test_data(listener=True)

    tokenize = TOKENIZERS[options.tokenizer]
    vec = SequenceVectorizer(unk_threshold=options.unk_threshold)

    print('Tokenizing training data')
    train_tokenized = [['<s>'] + tokenize(inst.input) + ['</s>'] for inst in train_insts]
    print('Tokenizing eval data')
    eval_tokenized = [['<s>'] + tokenize(inst.input) + ['</s>'] for inst in eval_insts]
    print('Initializing vectorizer')
    vec.add_all(train_tokenized)

    print_unk_ratio(train_tokenized, vec, 'Train')
    print_unk_ratio(eval_tokenized, vec, 'Eval')


def print_unk_ratio(tokenized, vec, name):
    print('')
    print(name + ':')
    unk_replaced = vec.unk_replace_all(tokenized)
    total_tokens = sum(len(s) for s in unk_replaced)
    num_unks = sum(s.count('<unk>') for s in unk_replaced)

    all_types = set()
    unk_types = set()
    for t, u in zip(tokenized, unk_replaced):
        assert len(t) == len(u), (t, u)
        for tw, uw in zip(t, u):
            all_types.add(tw)
            if uw == '<unk>':
                unk_types.add(tw)

    print('tokens: {}/{} ({:.2f}%)'.format(num_unks, total_tokens,
                                           num_unks * 100.0 / total_tokens))
    print('types:  {}/{} ({:.2f}%)'.format(len(unk_types), len(all_types),
                                           len(unk_types) * 100.0 / len(all_types)))


if __name__ == '__main__':
    count_unks()
