# Usage:
#   python print_wordvecs.py -C runs/modelname/config.json \
#                            --load runs/modelname/model.p \
#                            --wordvec_param_name custom_embeddings
import cPickle as pickle

import run_experiment  # NOQA: make sure we load all the command line args
from stanza.research import config

parser = config.get_options_parser()
parser.add_argument('--wordvec_param_name', default='desc_embed.W',
                    help='The name of the parameter containing the word vectors to be printed.')


def print_wordvecs(model, param_name):
    words = model.seq_vec.tokens
    import sys
    for param in model.params():
        if param.name != param_name:
            continue
        for word, row in zip(words, param.get_value()):
            print('\t'.join([word] + [repr(e) for e in row]))


if __name__ == '__main__':
    options = config.options(read=True)
    with open(options.load, 'rb') as infile:
        print_wordvecs(pickle.load(infile), options.wordvec_param_name)
