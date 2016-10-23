import cPickle as pickle
import json

import run_experiment  # NOQA: make sure we load all the command line args
from stanza.research import config
import color_instances


def output_sample(model):
    options = config.options()
    assert len(options.data_source) == 1, \
        'Only one data source at a time for sampling (got %s)' % options.data_source
    source = options.data_source[0]

    train_insts = color_instances.SOURCES[source].train_data(listener=options.listener)
    test_insts = color_instances.SOURCES[source].test_data(
        options.listener
    )[:options.test_size[0]]

    for output in model.predict(test_insts, random=True):
        print(json.dumps(output))


if __name__ == '__main__':
    options = config.options(read=True)
    with config.open('model.p', 'r') as infile:
        output_sample(pickle.load(infile))
