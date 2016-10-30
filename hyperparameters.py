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
import sys

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
parser.add_argument('--grid_search', default=None,
                    help='If not None, a path to a file containing parameter settings to use '
                         'as default in a grid search (try running all settings with '
                         'only one parameter different). The parameter settings should be '
                         'one line of an args file (space-separated values).')


def output_tuning_files(options):
    param_names, grid = load_grid(options.grid)
    if options.grid_search is None:
        with config.open('args', 'w') as argsfile:
            for i in range(options.num_samples):
                gpu_num = [str(i % 4)] if options.cycle_gpus else []
                argsfile.write(' '.join([random.choice(a) for a in grid] + gpu_num) + '\n')
    else:
        generate_grid_search(grid, options)
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


def generate_grid_search(grid, options):
    with open(options.grid_search, 'r') as settingsfile:
        base_settings = list(settingsfile)[0].split()
    search_configs = []
    gpu_num = 0
    for i, (base, values) in enumerate(zip(base_settings, grid)):
        try:
            base_index = values.index(base)
        except ValueError:
            print >>sys.stderr, 'Base setting "%s" not found in options %s' % (base, values)
        for j in range(len(values)):
            if values[j] != base:
                dist = abs(j - base_index)
                new_config = base_settings[:i] + [values[j]] + base_settings[i + 1:]
                if options.cycle_gpus:
                    new_config.append(str(gpu_num))
                    gpu_num = (gpu_num + 1) % options.cycle_gpus
                search_configs.append((dist, new_config))

    search_configs.sort(key=lambda p: p[0])

    with config.open('args', 'w') as argsfile:
        for dist_, new_config in search_configs:
            argsfile.write(' '.join(new_config) + '\n')


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
