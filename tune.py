# Usage:
#   python tune.py -C runs/template_run/config.json -T config/test_tune_config.hocon \
#                  -R runs/test_tune
#
#   [more arguments:]
#                  --tune_max_processes 16
#                  --tune_random 16
#                  --tune_local 32
#                  --tune_maximize -eval.perplexity.mean+eval.accuracy.mean+-eval.num_params
#
# See config/test_tune_config for format and options of the -T file.
import argparse
import itertools
import math
import multiprocessing as mp
import os
import Queue
import time

import pygtrie as trie
from stanza.research import config
from stanza.research.rng import get_rng

import run_experiment

parser = config.get_options_parser()
parser.add_argument('--tune_config', '-T',
                    help='Path to config file containing options for tuning search')
parser.add_argument('--tune_max_processes', type=int, default=16,
                    help='Number of processes to run at once for tuning search')
parser.add_argument('--tune_delay', type=float, default=10.0,
                    help='Number of seconds to wait in between starting each process. '
                         'If non-positive, do not wait.')
parser.add_argument('--tune_random', type=int, default=16,
                    help='Number of random parameter settings to try before switching '
                         'to local search')
parser.add_argument('--tune_local', type=int, default=0,
                    help='Maximum number of local search parameter settings to try.'
                         'If non-positive, keep searching until a local optimum is found.')
parser.add_argument('--tune_maximize', default='-eval.perplexity.gmean',
                    help='Name of result to optimize. Can be multiple results separated with a '
                         '"+" to add them. Prefix a result with "-" to minimize it.')

rng = get_rng()


def tune_queue(main_fn):
    config.redirect_output()
    options = config.options()
    with open(options.tune_config, 'r') as infile:
        tune_options = config.HoconConfigFileParser().parse(infile)

    reg = ProcessRegistry(main_fn, tune_options, options.tune_maximize)

    remaining_random = options.tune_random
    remaining_local = options.tune_local
    if options.tune_local <= 0:
        remaining_local = None

    try:
        reg.start_default()
        while remaining_random > 0 and reg.running_processes < options.tune_max_processes:
            reg.start_random()
            remaining_random -= 1

        while remaining_local > 0 and reg.running_processes < options.tune_max_processes:
            reg.start_local()
            remaining_random -= 1

        while reg.running_processes > 0:
            name, objective = reg.get()
            print('\nTUNE: {:10.3f} {}\n'.format(objective, name[:70]))

            while remaining_random > 0 and reg.running_processes < options.tune_max_processes:
                reg.start_random()
                remaining_random -= 1

            while (remaining_local is None or remaining_local > 0) and \
                    reg.running_processes < options.tune_max_processes:
                try:
                    reg.start_local()
                    if remaining_local is not None:
                        remaining_local -= 1
                except StopIteration:
                    print('no new local search candidates')
                    break
    except KeyboardInterrupt:
        reg.terminate()

    print('')
    print('best result:')
    print('{:10.3f} {}'.format(reg.best_objective, str(reg.best_name)[:70]))


class ProcessRegistry(object):
    def __init__(self, main_fn, tune_options, objective):
        self.main_fn = main_fn
        self.tune_options = tune_options
        self.objective = objective

        self.abbreviations = abbreviate(tune_options['options'].keys())

        self.proc_for_name = {}
        self.name_for_options = {}
        self.options_for_name = {}

        self.best_objective = None
        self.best_name = None

        self.running_processes = 0

        self.results_queue = mp.Queue()

    def start_default(self):
        self.start(self.base_options(), mode='default')

    def start_random(self):
        self.start(self.generate_new_random_options(), mode='random')

    def start_local(self):
        self.start(self.generate_new_local_options(), mode='local')

    def start(self, tuned_options, mode='manual'):
        name = self.short_name(tuned_options)
        options_dict = dict(config.options().__dict__)
        options_dict['run_dir'] = os.path.join(options_dict['run_dir'], name)
        options_dict['overwrite'] = False
        options_dict['config'] = None
        for k, v in tuned_options:
            options_dict[k] = v
        options = argparse.Namespace(**options_dict)

        if options_dict['tune_delay'] > 0:
            time.sleep(options_dict['tune_delay'])
        proc = mp.Process(target=queue_results,
                          args=(self.main_fn, options, name, self.results_queue))
        self.proc_for_name[name] = proc
        self.name_for_options[tuned_options] = name
        self.options_for_name[name] = tuned_options
        self.running_processes += 1
        print('starting {}: {}'.format(mode, name))
        proc.start()

    def get(self):
        name, results = self.results_queue.get()

        try:
            self.proc_for_name[name].join(timeout=10)
        except Queue.Empty:
            self.proc_for_name[name].terminate()
        self.proc_for_name[name] = None
        self.running_processes -= 1

        objective = get_objective(results, self.objective)
        if self.best_objective is None or self.best_objective < objective:
            self.best_objective = objective
            self.best_name = name

        return name, objective

    def terminate(self):
        running_names = []
        for name, proc in self.proc_for_name.iteritems():
            running_names.append(name)
            if proc is not None:
                proc.terminate()
        for name in running_names:
            self.proc_for_name[name] = None
            self.running_processes -= 1

    def generate_new_random_options(self):
        while True:
            tuned_options_list = []
            for key, values in self.tune_options['options'].iteritems():
                tuned_options_list.append((key, values[rng.randint(len(values))]))
            tuned_options = tuple(tuned_options_list)
            if tuned_options not in self.name_for_options:
                return tuned_options

    def generate_new_local_options(self):
        if self.best_name is None:
            base = self.base_options()
        else:
            base = self.options_for_name[self.best_name]

        unexplored_neighbors = []
        for k, v_orig in base:
            changes = [
                [(k, v_new) for v_new in self.tune_options['options'][k] if v_new != v_orig]
            ]
            if k in self.tune_options['interactions']:
                for k2 in self.tune_options['interactions'][k]:
                    changes.append(
                        [(k2, v_new) for v_new in self.tune_options['options'][k2]]
                    )
            for changeset in itertools.product(*changes):
                changeset = dict(changeset)
                neighbor = tuple(
                    (k2, (changeset[k2] if k2 in changeset else v))
                    for k2, v in base
                )
                if neighbor not in self.name_for_options:
                    unexplored_neighbors.append(neighbor)

        if unexplored_neighbors:
            return unexplored_neighbors[rng.randint(len(unexplored_neighbors))]
        else:
            raise StopIteration

    def base_options(self):
        options = config.options()
        return tuple((key, getattr(options, key))
                     for key, _ in self.tune_options['options'].iteritems())

    def short_name(self, tuned_options):
        return '+'.join(
            '{}={}'.format(self.abbreviations[k], v)
            for k, v in tuned_options
        )


def abbreviate(all_keys):
    '''
    Find a minimal abbreviation for each key in a set of string keys.
    This is the series of character decisions you have to make to reach
    that key, or, perhaps more familiar, the series of characters you
    would have to type if you hit 'tab' before each character to
    auto-complete any unique prefix.

    >>> abbreviate(['spe_lea_rate', 'spe_cell_size', 'spe_opt'])['spe_lea_rate']
    'l'
    >>> abbreviate(['spe_lea_rate', 'ste_lea_rate'])['spe_lea_rate']
    'p'
    >>> abbreviate(['spe_lea_rate', 'spe_lea_rates'])['spe_lea_rate']
    ''
    >>> abbreviate(['spe_lea_rate', 'spe_lea_rates'])['spe_lea_rates']
    's'
    '''
    t = trie.CharTrie({k: True for k in all_keys})

    # Add a node for each string that is a branching point in the trie
    internal_nodes = {}

    def collect_internal_nodes(path_conv, path, children, value=None):
        children = list(children)
        if len(children) > 1:
            internal_nodes[path_conv(path)] = True

    t.traverse(collect_internal_nodes)
    t.update(internal_nodes)

    abbrev_map = {}
    for key in all_keys:
        # t.prefixes(key) gives all branching points on the path leading to key.
        # Extract the next character after each branching point and concatenate them.
        abbrev_map[key] = ''.join(
            key[len(prefix):len(prefix) + 1]
            for prefix, _ in t.prefixes(key)
        )

    return abbrev_map


def get_objective(results, spec):
    if isinstance(results, Exception):
        return float('-inf')
    total = 0.0
    pieces = spec.split('+')
    for piece in pieces:
        if piece.startswith('-'):
            total -= results[piece[1:]]
        else:
            total += results[piece]
    if math.isnan(total):
        # Any nan is worse than any finite number
        total = float('-inf')
    return total


def test_main():
    options = config.options()
    import sys
    print('stdout')
    sys.stderr.write('stderr\n')

    return {}, {
        'eval.perplexity.gmean': (options.speaker_learning_rate +
                                  options.speaker_cell_size +
                                  len(options.speaker_optimizer))
    }


def queue_results(main_fn, options, name, q):
    try:
        # Create run dir and force options to become the passed Namespace object
        config.set_options(options)
        config.redirect_output()
        (train, test) = main_fn()
        results = dict(train)
        results.update(test)
        q.put((name, results))
    except Exception as e:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        q.put((name, e))


if __name__ == '__main__':
    tune_queue(run_experiment.main)
