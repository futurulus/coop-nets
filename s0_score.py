# Build a grids file for score from the S0 model rather than the L0 model.
# Usage:
#   python s0_score.py -C runs/s0/config.json --load runs/s0/model.p \
#                      --grids_file runs/lstar/grids.0.jsons.gz
# Output will be written to runs/lstar/s0_grids.0.jsons.gz

import gzip
import json
import numpy as np
import cPickle as pickle
import os

import run_experiment  # NOQA: make sure we load all the command line args
from stanza.research import config, instance
from stanza.monitoring import progress

parser = config.get_options_parser()
parser.add_argument('--grids_file', help='Path to input grids.*.jsons.gz file. '
                                         'Also used to determine the output file path. ')


def output_grids(model, input_filename):
    with gzip.open(input_filename, 'rb') as infile:
        grids = [json.loads(line.strip()) for line in infile]

    dirname, filename = os.path.split(input_filename)

    data_filename = os.path.join(dirname, 'data.eval.jsons')
    with open(data_filename, 'r') as infile:
        insts = [json.loads(line.strip()) for line in infile]

    output_filename = os.path.join(dirname, 's0_' + filename)
    with gzip.open(output_filename, 'w') as outfile:
        progress.start_task('Example', len(insts))
        for i, (inst, grid) in enumerate(zip(insts, grids)):
            progress.progress(i)
            insts, shape = build_insts(inst, grid)
            scores = model.score(insts, verbosity=-4)
            substitute_grid(scores, grid, shape)
            json.dump(grid, outfile)
            outfile.write('\n')
        progress.end_task()


def build_insts(orig_inst, grid):
    # num_sample_sets * context_len * num_alt_utts
    num_sample_sets = len(grid['sets'])
    context_len = len(orig_inst['alt_outputs'])
    num_alt_utts = len(grid['sets'][0]['utts'])

    new_insts = []
    for s, ss in enumerate(grid['sets']):
        for t in range(len(orig_inst['alt_outputs'])):
            for u, utt in enumerate(ss['utts']):
                new_insts.append(instance.Instance(input=t,
                                                   alt_inputs=orig_inst['alt_outputs'],
                                                   output=utt))
    return new_insts, (num_sample_sets, context_len, num_alt_utts)


def substitute_grid(scores, grid, shape):
    scores_array = np.array(scores).reshape(shape)
    for s, ss in enumerate(grid['sets']):
        ss['S0'] = scores_array[s, :, :].tolist()


if __name__ == '__main__':
    options = config.options(read=True)
    with open(options.load, 'rb') as infile:
        model = pickle.load(infile)
    output_grids(model, options.grids_file)
