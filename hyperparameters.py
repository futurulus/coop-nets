r'''
Generate files for running random search on GNU Parallel. Example usage:

    python hyperparameters.py --run_dir runs/tune \
                              --grid runs/tune/grid.tsv

This script will create a file called `args` containing the hyperparameter
settings and a file called `command` containing a GNU Parallel invocation
that loads those settings. The files will be created in the `run_dir`.
'''
import os
import random

from stanza.research import config

parser = config.get_options_parser()
parser.add_argument('--num_samples', type=int, default=200,
                    help='For random search, number of hyperparameter settings to sample.')
parser.add_argument('--num_processes', type=int, default=8,
                    help='Number of simultaneous jobs to run (-j argument to parallel).')
parser.add_argument('--cycle_gpus', type=int, default=0,
                    help='If nonzero, rotate jobs between this number of GPUs, in the order '
                         'the jobs appear in the hyperparameter settings list. In other words, '
                         'if cycle_gpus=2, the first job will be put on --device gpu1, the '
                         'second on --device gpu2, the third on --device gpu1, and so forth.')
parser.add_argument('--grid', default=None,
                    help='Path to a tab-separated value file containing parameter names in '
                         'the first column, followed by possible values for that parameter '
                         '(as strings).')


def output_tuning_files(options):
    param_names, grid = load_grid(options.grid)
    with config.open('args', 'w') as argsfile:
        for i in range(options.num_samples):
            gpu_num = [str(i % 4)] if options.cycle_gpus else []
            argsfile.write(' '.join([random.choice(a) for a in grid] + gpu_num) + '\n')
    with config.open('command', 'w') as commandfile:
        commandfile.write("parallel -j%d --delay 30 --tagstring " % options.num_processes)
        commandfile.write(build_tag_string(param_names))
        commandfile.write(" --eta --colsep ' ' echo python run_experiment.py ")
        for i, param_name in enumerate(param_names):
            if param_name.startswith('*'):
                param_name = param_name[1:]
            commandfile.write(" --%s {%d}" % (param_name, i + 1))
        if options.cycle_gpus:
            commandfile.write(" --device gpu{%d}" % (len(param_names) + 1))
        commandfile.write(" --run_dir %s" % build_run_dir(options.run_dir, param_names, grid))
        commandfile.write(" :::: %s\n" % os.path.join(options.run_dir, 'args'))


def load_grid(filename):
    param_names = []
    grid = []
    with open(filename, 'r') as infile:
        for line in infile:
            if not line.strip():
                continue
            cols = line[:-1].split('\t')
            param_names.append(cols[0])
            grid.append(cols[1:])
    return param_names, grid


def build_tag_string(param_names):
    index_strings = []
    for i, name in enumerate(param_names):
        if name.startswith('*'):
            index_strings.append('{%d}' % (i + 1))
    return '-'.join(index_strings)


def build_run_dir(run_dir, param_names, grid):
    index_strings = []
    for i, (name, values) in enumerate(zip(param_names, grid)):
        if len(values) == 1:
            continue
        index_strings.append('{%d/}' % (i + 1))
    return os.path.join(run_dir, '-'.join(index_strings))


if __name__ == '__main__':
    options = config.options(read=True)
    output_tuning_files(options)
