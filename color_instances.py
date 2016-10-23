import colorsys
import csv
from collections import namedtuple, defaultdict
import json

try:
    from rugstk.data.munroecorpus import munroecorpus
except ImportError:
    import sys
    sys.stderr.write('Munroe corpus not found (have you run '
                     './dependencies?)\n')
    raise

from stanza.research import config
from stanza.research.instance import Instance
from stanza.research.rng import get_rng
from colorutils import hsl_to_hsv

import tuna_instances


rng = get_rng()

parser = config.get_options_parser()
parser.add_argument('--num_distractors', type=int, default=4,
                    help='The number of random colors to include in addition to the true '
                         'color in generating reference game instances. Ignored if not '
                         'using one of the `ref_` data sources.')
parser.add_argument('--train_data_file', type=str, default=None,
                    help='Path to a json file to use as the training dataset. Ignored if '
                         'not using the `file` data source.')
parser.add_argument('--test_data_file', type=str, default=None,
                    help='Path to a json file to use as the evaluation dataset. Ignored if '
                         'not using the `file` data source.')


def load_colors(h, s, v):
    return zip(*(munroecorpus.open_datafile(d) for d in [h, s, v]))


def get_training_instances(listener=False):
    h, s, v = munroecorpus.get_training_handles()
    insts = [
        (Instance(input=name,
                  output=color)
         if listener else
         Instance(input=color,
                  output=name))
        for name in h
        for color in load_colors(h[name], s[name], v[name])
    ]
    rng.shuffle(insts)
    return insts


def get_eval_instances(handles, listener=False):
    insts = [
        (Instance(input=name,
                  output=tuple(color))
         if listener else
         Instance(input=tuple(color),
                  output=name))
        for name, handle in handles.iteritems()
        for color in munroecorpus.open_datafile(handle)
    ]
    rng.shuffle(insts)
    return insts


def get_dev_instances(listener=False):
    return get_eval_instances(munroecorpus.get_dev_handles(), listener=listener)


def get_test_instances(listener=False):
    return get_eval_instances(munroecorpus.get_test_handles(), listener=listener)


def tune_train(listener=False):
    all_train = get_training_instances(listener=listener)
    return all_train[:-100000]


def tune_test(listener=False):
    all_train = get_training_instances(listener=listener)
    return all_train[-100000:]


def json_file_train(listener='ignored'):
    options = config.options()
    with open(options.train_data_file, 'r') as infile:
        dataset = [json.loads(line.strip()) for line in infile]
    return [Instance(**d) for d in dataset]


def json_file_test(listener='ignored'):
    options = config.options()
    with open(options.test_data_file, 'r') as infile:
        dataset = [json.loads(line.strip()) for line in infile]
    return [Instance(**d) for d in dataset]


def pairs_to_insts(data, listener=False):
    return [
        (Instance(input=name, output=color)
         if listener else
         Instance(input=color, output=name))
        for name, color in data
    ]


def triples_to_insts(data, listener=False):
    return [
        (Instance(input=name, output=color, alt_outputs=context)
         if listener else
         Instance(input=color, alt_inputs=context, output=name))
        for name, color, context in data
    ]


def empty_str(listener=False):
    data = [('', (167.74193548404, 96.3730569948, 75.6862745098))]
    return pairs_to_insts(data, listener=listener)


def one_word(listener=False):
    data = [('green', (167.74193548404, 96.3730569948, 75.6862745098))]
    return pairs_to_insts(data, listener=listener)


def two_word(listener=False):
    data = [('shocking green', (167.74193548404, 96.3730569948, 75.6862745098))]
    return pairs_to_insts(data, listener=listener)


def scalar_imp_train(listener=False):
    data = [
        ('blue', (240., 100., 100.)),
        ('blue', (180., 100., 100.)),
        ('teal', (180., 100., 100.)),
    ]
    return pairs_to_insts(data, listener=listener)


def scalar_imp_test(listener=False):
    data = [
        ('blue', (240., 100., 100.)),
        ('blue', (180., 100., 100.)),
        ('teal', (240., 100., 100.)),
        ('teal', (180., 100., 100.)),
    ]
    return pairs_to_insts(data, listener=listener)


def scalar_imp_level2_train(listener=False):
    data = [
        ('blue', (240., 100., 100.)),
        ('blue', (170., 100., 70.)),
        ('green', (170., 100., 70.)),
        ('green', (80., 100., 100.)),
        ('yellow', (80., 100., 100.)),
    ]
    return pairs_to_insts(data, listener=listener)


def scalar_imp_level2_test(listener=False):
    data = [
        ('blue', (240., 100., 100.)),
        ('blue', (170., 100., 70.)),
        ('blue', (80., 100., 100.)),
        ('green', (240., 100., 100.)),
        ('green', (170., 100., 70.)),
        ('green', (80., 100., 100.)),
        ('yellow', (240., 100., 100.)),
        ('yellow', (170., 100., 70.)),
        ('yellow', (80., 100., 100.)),
    ]
    return pairs_to_insts(data, listener=listener)


def amsterdam_literal_train(listener=False):
    data = [
        ('light purple', 0, [(260., 45., 100.), (260., 100., 100.)]),
        ('purple', 0, [(260., 45., 100.), (260., 100., 100.)]),
        ('light', 0, [(260., 45., 100.), (260., 100., 100.)]),
        ('', 0, [(260., 45., 100.), (260., 100., 100.)]),
        ('light purple', 1, [(260., 100., 100.), (260., 45., 100.)]),
        ('purple', 1, [(260., 100., 100.), (260., 45., 100.)]),
        ('light', 1, [(260., 100., 100.), (260., 45., 100.)]),
        ('', 1, [(260., 100., 100.), (260., 45., 100.)]),
        ('pinkish purple', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('pinkish', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('pinkish purple', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('pinkish', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('', 0, [(300., 100., 100.), (260., 100., 100.)]),
    ]
    return triples_to_insts(data, listener=listener)


def amsterdam_typical_train(listener=False):
    data = [
        ('light purple', 0, [(260., 45., 100.), (260., 100., 100.)]),
        ('purple', 0, [(260., 45., 100.), (260., 100., 100.)]),
        ('purple', 1, [(260., 45., 100.), (260., 100., 100.)]),
        ('purple', 1, [(260., 45., 100.), (260., 100., 100.)]),
        ('light purple', 1, [(260., 100., 100.), (260., 45., 100.)]),
        ('purple', 1, [(260., 100., 100.), (260., 45., 100.)]),
        ('purple', 0, [(260., 100., 100.), (260., 45., 100.)]),
        ('purple', 0, [(260., 100., 100.), (260., 45., 100.)]),
    ] * 3 + [
        ('pinkish purple', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('pinkish', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 0, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 0, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 0, [(260., 100., 100.), (300., 100., 100.)]),
        ('pinkish purple', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('pinkish', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 1, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 1, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 1, [(300., 100., 100.), (260., 100., 100.)]),
    ] * 2
    return triples_to_insts(data, listener=listener)


def amsterdam_1word_train(listener=False):
    data = [
        ('light', 0, [(260., 45., 100.), (260., 100., 100.)]),
        ('purple', 0, [(260., 45., 100.), (260., 100., 100.)]),
        ('purple', 1, [(260., 45., 100.), (260., 100., 100.)]),
        ('purple', 1, [(260., 45., 100.), (260., 100., 100.)]),
        ('light', 1, [(260., 100., 100.), (260., 45., 100.)]),
        ('purple', 1, [(260., 100., 100.), (260., 45., 100.)]),
        ('purple', 0, [(260., 100., 100.), (260., 45., 100.)]),
        ('purple', 0, [(260., 100., 100.), (260., 45., 100.)]),
    ] * 3 + [
        ('pinkish', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('pinkish', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 0, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 0, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 0, [(260., 100., 100.), (300., 100., 100.)]),
        ('pinkish', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('pinkish', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 1, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 1, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 1, [(300., 100., 100.), (260., 100., 100.)]),
    ] * 2
    return triples_to_insts(data, listener=listener)


def amsterdam_unambiguous_train(listener=False):
    data = [
        ('light purple', 0, [(260., 45., 100.), (260., 100., 100.)]),
        ('purple', 1, [(260., 45., 100.), (260., 100., 100.)]),
        ('light purple', 1, [(260., 100., 100.), (260., 45., 100.)]),
        ('purple', 0, [(260., 100., 100.), (260., 45., 100.)]),
    ] * 3 + [
        ('pinkish purple', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('pinkish', 1, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 0, [(260., 100., 100.), (300., 100., 100.)]),
        ('purple', 0, [(260., 100., 100.), (300., 100., 100.)]),
        ('pinkish purple', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('pinkish', 0, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 1, [(300., 100., 100.), (260., 100., 100.)]),
        ('purple', 1, [(300., 100., 100.), (260., 100., 100.)]),
    ] * 2
    return triples_to_insts(data, listener=listener)


def amsterdam_test(listener=False):
    data = [
        ('', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('light', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('pinkish', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('purple', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('light purple', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('pinkish purple', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('light pinkish', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('light pinkish purple', 0, [(300., 45., 100.), (300., 100., 100.)]),
    ]
    return triples_to_insts(data, listener=listener)


def amsterdam_test_allways(listener=False):
    data = [
        ('', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('light', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('pinkish', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('purple', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('light purple', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('pinkish purple', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('light pinkish', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('light pinkish purple', 0, [(300., 45., 100.), (300., 100., 100.)]),
        ('', 0, [(300., 100., 100.), (300., 45., 100.)]),
        ('light', 0, [(300., 100., 100.), (300., 45., 100.)]),
        ('pinkish', 0, [(300., 100., 100.), (300., 45., 100.)]),
        ('purple', 0, [(300., 100., 100.), (300., 45., 100.)]),
        ('light purple', 0, [(300., 100., 100.), (300., 45., 100.)]),
        ('pinkish purple', 0, [(300., 100., 100.), (300., 45., 100.)]),
        ('light pinkish', 0, [(300., 100., 100.), (300., 45., 100.)]),
        ('light pinkish purple', 0, [(300., 100., 100.), (300., 45., 100.)]),
        ('', 1, [(300., 45., 100.), (300., 100., 100.)]),
        ('light', 1, [(300., 45., 100.), (300., 100., 100.)]),
        ('pinkish', 1, [(300., 45., 100.), (300., 100., 100.)]),
        ('purple', 1, [(300., 45., 100.), (300., 100., 100.)]),
        ('light purple', 1, [(300., 45., 100.), (300., 100., 100.)]),
        ('pinkish purple', 1, [(300., 45., 100.), (300., 100., 100.)]),
        ('light pinkish', 1, [(300., 45., 100.), (300., 100., 100.)]),
        ('light pinkish purple', 1, [(300., 45., 100.), (300., 100., 100.)]),
        ('', 1, [(300., 100., 100.), (300., 45., 100.)]),
        ('light', 1, [(300., 100., 100.), (300., 45., 100.)]),
        ('pinkish', 1, [(300., 100., 100.), (300., 45., 100.)]),
        ('purple', 1, [(300., 100., 100.), (300., 45., 100.)]),
        ('light purple', 1, [(300., 100., 100.), (300., 45., 100.)]),
        ('pinkish purple', 1, [(300., 100., 100.), (300., 45., 100.)]),
        ('light pinkish', 1, [(300., 100., 100.), (300., 45., 100.)]),
        ('light pinkish purple', 1, [(300., 100., 100.), (300., 45., 100.)]),
    ]
    return triples_to_insts(data, listener=listener)


def reference_game_train(gen_func):
    def generate_refgame_train(listener=False):
        return reference_game(get_training_instances(listener=listener),
                              gen_func, listener=listener)
    return generate_refgame_train


def reference_game_test(gen_func):
    def generate_refgame_test(listener=False):
        return reference_game(get_dev_instances(listener=listener),
                              gen_func, listener=listener)
    return generate_refgame_test


def reference_game(insts, gen_func, listener=False):
    options = config.options()
    for i in range(len(insts)):
        color = insts[i].output if listener else insts[i].input
        distractors = [gen_func(color) for _ in range(options.num_distractors)]
        answer = rng.randint(0, len(distractors) + 1)
        context = distractors[:answer] + [color] + distractors[answer:]
        ref_inst = (Instance(insts[i].input, answer, alt_outputs=context)
                    if listener else
                    Instance(answer, insts[i].output, alt_inputs=context))
        insts[i] = ref_inst
    return insts


def hawkins_context(listener=False, suffix=''):
    messages = defaultdict(list)
    with open('hawkins_data/colorReferenceMessage%s.csv' % suffix, 'r') as infile:
        for row in csv.DictReader(infile):
            if row['sender'] == 'speaker':
                message = row['contents']  # TODO: clean, tokenize?
                messages[(row['gameid'], row['roundNum'])].append(message)

    result = []
    with open('hawkins_data/colorReferenceClicks%s.csv' % suffix, 'r') as infile:
        reader = csv.DictReader(infile)
        seen = set()
        for row in reader:
            key = (row['gameid'], row['roundNum'])
            if key in seen:
                # print('Duplicate key: %s' % (key,))
                continue
            seen.add(key)
            context = [
                (hsl_to_hsv((row['%sColH' % i],
                             row['%sColS' % i],
                             row['%sColL' % i])),
                 row['%sLocS' % i], row['%sStatus' % i])
                for i in ('click', 'alt1', 'alt2')
            ]
            context.sort(key=lambda c: c[1])
            target_idx = [i for i, (_, _, status) in enumerate(context) if status == 'target']
            assert len(target_idx) == 1, context
            target_idx = target_idx[0]
            alt_colors = [c for (c, _, _) in context]
            message = ' ~ '.join(messages[key])

            if listener:
                inst = Instance(input=message, output=target_idx, alt_outputs=alt_colors,
                                source=key)
            else:
                inst = Instance(input=target_idx, alt_inputs=alt_colors, output=message,
                                source=key)
            result.append(inst)
    return result


def hawkins_target(listener=False):
    insts = hawkins_context(listener=listener)
    return [(Instance(output=inst.alt_outputs[inst.output], input=inst.input,
                      source=inst.__dict__)
             if listener else
             Instance(input=inst.alt_inputs[inst.input], output=inst.output,
                      source=inst.__dict__))
            for inst in insts]


def hawkins_train(listener=False, suffix=''):
    insts = hawkins_context(listener=listener, suffix=suffix)
    num_insts = len(insts) / 3
    train_insts = insts[:num_insts]
    rng.shuffle(train_insts)
    return train_insts


def hawkins_dev(listener=False, suffix=''):
    insts = hawkins_context(listener=listener, suffix=suffix)
    num_insts = len(insts) / 3
    dev_insts = insts[num_insts:num_insts * 2]
    return dev_insts


def hawkins_test(listener=False, suffix=''):
    insts = hawkins_context(listener=listener, suffix=suffix)
    num_insts = len(insts) / 3
    train_insts = insts[num_insts * 2:]
    return train_insts


def hawkins_tune_train(listener=False, suffix='', tuning_insts=350):
    insts = hawkins_context(listener=listener, suffix=suffix)
    num_insts = len(insts) / 3
    train_insts = insts[:num_insts - tuning_insts]
    rng.shuffle(train_insts)
    return train_insts


def hawkins_tune_test(listener=False, suffix='', tuning_insts=350):
    insts = hawkins_context(listener=listener, suffix=suffix)
    num_insts = len(insts) / 3
    tune_insts = insts[num_insts - tuning_insts:num_insts]
    return tune_insts


def hawkins_big_train(listener=False):
    return hawkins_train(listener=listener, suffix='2')


def hawkins_big_dev(listener=False):
    return hawkins_dev(listener=listener, suffix='2')


def hawkins_big_test(listener=False):
    return hawkins_test(listener=listener, suffix='2')


def hawkins_big_tune_train(listener=False):
    return hawkins_tune_train(listener=listener, suffix='2', tuning_insts=3500)


def hawkins_big_tune_test(listener=False):
    return hawkins_tune_test(listener=listener, suffix='2', tuning_insts=3500)


def uniform(color):
    r, g, b = rng.uniform(size=(3,))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0, s * 100.0, v * 100.0


def uniform_int(color):
    r, g, b = rng.uniform(size=(3,))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 360.0), int(round(s * 101.0 - 0.5)), int(round(v * 101.0 - 0.5))


def linear_rgb(color):
    coord = rng.randint(0, 3)
    val = rng.uniform()
    h, s, v = color
    result = list(colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0))
    result[coord] = val
    h, s, v = colorsys.rgb_to_hsv(*result)
    return h * 360.0, s * 100.0, v * 100.0


def linear_hsv(color):
    coord = rng.randint(0, 3)
    val = rng.uniform()
    h, s, v = color
    result = [h / 360.0, s / 100.0, v / 100.0]
    result[coord] = val
    h, s, v = result
    return h * 360.0, s * 100.0, v * 100.0


DataSource = namedtuple('DataSource', ['train_data', 'test_data'])

SOURCES = {
    'dev': DataSource(get_training_instances, get_dev_instances),
    'test': DataSource(get_training_instances, get_test_instances),
    'tune': DataSource(tune_train, tune_test),
    'file': DataSource(json_file_train, json_file_test),
    '2word': DataSource(two_word, two_word),
    '1word': DataSource(one_word, one_word),
    '0word': DataSource(empty_str, empty_str),
    'scalar': DataSource(scalar_imp_train, scalar_imp_test),
    'scalar_lv2': DataSource(scalar_imp_level2_train, scalar_imp_level2_test),
    'ref_uni': DataSource(reference_game_train(uniform), reference_game_test(uniform)),
    'ref_linrgb': DataSource(reference_game_train(linear_rgb), reference_game_test(linear_rgb)),
    'ref_linhsv': DataSource(reference_game_train(linear_hsv), reference_game_test(linear_hsv)),
    'hawkins': DataSource(lambda listener: [], hawkins_context),
    'hawkins_target': DataSource(lambda listener: [], hawkins_target),
    'hawkins_dev': DataSource(hawkins_train, hawkins_dev),
    'hawkins_test': DataSource(hawkins_train, hawkins_test),
    'hawkins_tune': DataSource(hawkins_tune_train, hawkins_tune_test),
    'ams_literal': DataSource(amsterdam_literal_train, amsterdam_test),
    'ams_unambig': DataSource(amsterdam_unambiguous_train, amsterdam_test),
    'ams_1word': DataSource(amsterdam_1word_train, amsterdam_test),
    'ams_typical': DataSource(amsterdam_typical_train, amsterdam_test),
    'ams_typical_allways': DataSource(amsterdam_typical_train, amsterdam_test_allways),
    'tuna_cv': DataSource(tuna_instances.tuna_train_cv, tuna_instances.tuna_test_cv),
    'tuna_dev': DataSource(tuna_instances.tuna08_train, tuna_instances.tuna08_dev),
    'tuna_test': DataSource(tuna_instances.tuna08_train, tuna_instances.tuna08_test)
}
