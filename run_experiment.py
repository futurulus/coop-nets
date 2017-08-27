from stanza.research import config
config.redirect_output()

from stanza.cluster import pick_gpu
parser = config.get_options_parser()
parser.add_argument('--device', default=None,
                    help='The device to use in Theano ("cpu" or "gpu[0-n]"). If None, '
                         'pick a free-ish device automatically.')
options, extras = parser.parse_known_args()
if '-h' in extras or '--help' in extras:
    # If user is just asking for the options, don't scare them
    # by saying we're picking a GPU...
    pick_gpu.bind_theano('cpu')
else:
    pick_gpu.bind_theano(options.device)


from stanza.monitoring import progress
from stanza.research import evaluate, metrics, output
import datetime
import numbers
import learners
import color_instances

parser.add_argument('--learner', default='Histogram', choices=learners.LEARNERS.keys(),
                    help='The name of the model to use in the experiment.')
parser.add_argument('--load', metavar='MODEL_FILE', default=None,
                    help='If provided, skip training and instead load a pretrained model '
                         'from the specified path. If None or an empty string, train a '
                         'new model.')
parser.add_argument('--train_size', type=int, default=[-1], nargs='+',
                    help='The number of examples to use in training. This number should '
                         '*include* examples held out for validation. If None or negative, use the '
                         'whole training set.')
parser.add_argument('--validation_size', type=int, default=[0], nargs='+',
                    help='The number of examples to hold out from the training set for '
                         'monitoring generalization error.')
parser.add_argument('--test_size', type=int, default=[-1], nargs='+',
                    help='The number of examples to use in testing. '
                         'If None or negative, use the whole dev/test set.')
parser.add_argument('--data_source', default=['dev'], nargs='+',
                    choices=color_instances.SOURCES.keys(),
                    help='The type of data to use. Can supply several for sequential training.')
parser.add_argument('--output_train_data', type=config.boolean, default=False,
                    help='If True, write out the training dataset (after cutting down to '
                         '`train_size`) as a JSON-lines file in the output directory.')
parser.add_argument('--output_test_data', type=config.boolean, default=False,
                    help='If True, write out the evaluation dataset (after cutting down to '
                         '`test_size`) as a JSON-lines file in the output directory.')
parser.add_argument('--listener', type=config.boolean, default=False,
                    help='If True, evaluate on listener accuracy (description -> color). '
                         'Otherwise evaluate on speaker accuracy (color -> description).')
parser.add_argument('--progress_tick', type=int, default=300,
                    help='The number of seconds between logging progress updates.')


def main():
    options = config.options()

    progress.set_resolution(datetime.timedelta(seconds=options.progress_tick))

    train_datasets = []
    validation_datasets = []
    test_datasets = []

    if len(options.train_size) == 1:
        options.train_size = options.train_size * len(options.data_source)
    else:
        assert len(options.train_size) == len(options.data_source)
    if len(options.validation_size) == 1:
        options.validation_size = options.validation_size * len(options.data_source)
    else:
        assert len(options.validation_size) == len(options.data_source)
    if len(options.test_size) == 1:
        options.test_size = options.test_size * len(options.data_source)
    else:
        assert len(options.test_size) == len(options.data_source)

    for source, train_size, validation_size, test_size in zip(options.data_source,
                                                              options.train_size,
                                                              options.validation_size,
                                                              options.test_size):
        if train_size < 0:
            train_size = None
        train_insts = color_instances.SOURCES[source].train_data(
            listener=options.listener
        )[:train_size]
        if validation_size:
            assert validation_size < len(train_insts), \
                ('No training data after validation split! (%d <= %d)' %
                 (len(train_insts), validation_size))
            validation_insts = train_insts[-validation_size:]
            validation_datasets.append(validation_insts)
            train_insts = train_insts[:-validation_size]
        else:
            validation_datasets.append(None)
        train_datasets.append(train_insts)

        if test_size < 0:
            test_size = None
        test_insts = color_instances.SOURCES[source].test_data(
            options.listener
        )[:test_size]
        test_datasets.append(test_insts)

    learner = learners.new(options.learner)

    m = [metrics.log_likelihood,
         metrics.log_likelihood_bits,
         metrics.perplexity,
         metrics.aic]
    example_inst = get_example_inst(test_datasets, train_datasets)
    if options.listener and not isinstance(example_inst.output, numbers.Integral):
        m.append(metrics.squared_error)
    elif isinstance(example_inst.output, (tuple, list)):
        m.append(metrics.prec1)
        if example_inst.output and isinstance(example_inst.output, basestring):
            m.extend([metrics.bleu, metrics.token_perplexity_macro, metrics.token_perplexity_micro])
    else:
        m.append(metrics.accuracy)
        if example_inst.output and isinstance(example_inst.output, basestring):
            m.extend([metrics.bleu, metrics.token_perplexity_macro, metrics.token_perplexity_micro])

    multi_train = (len(options.data_source) > 1)
    if options.load:
        with open(options.load, 'rb') as infile:
            learner.load(infile)
    else:
        if hasattr(learner, '_data_to_arrays'):
            # XXX: is there a better way to ensure that the vocabulary is defined
            # before training starts?
            for train_insts in train_datasets[1:]:
                learner._data_to_arrays(train_insts, init_vectorizer=True)

        for i, (source, train_insts, validation_insts) in enumerate(zip(options.data_source,
                                                                        train_datasets,
                                                                        validation_datasets)):
            if not train_insts:
                continue

            if i > 0:
                learner.train(train_insts, validation_insts, metrics=m, keep_params=True)
            else:
                learner.train(train_insts, validation_insts, metrics=m)
            with open(config.get_file_path('model.p'), 'wb') as outfile:
                learner.dump(outfile)

            if multi_train:
                split_id = 'train_' + source
            else:
                split_id = 'train'
            train_results = evaluate.evaluate(learner, train_insts, metrics=m, split_id=split_id,
                                              write_data=options.output_train_data)
            output.output_results(train_results, split_id)

    for i, (source, test_insts) in enumerate(zip(options.data_source,
                                                 test_datasets)):
        if not test_insts:
            continue
        if multi_train:
            split_id = 'eval_' + source
        else:
            split_id = 'eval'
        test_results = evaluate.evaluate(learner, test_insts, metrics=m, split_id=split_id,
                                         write_data=options.output_test_data)
        output.output_results(test_results, split_id)


def get_example_inst(test_datasets, train_datasets):
    # Use test if any are nonempty, if not, back off to train
    for dataset in test_datasets:
        if dataset:
            return dataset[0]
    for dataset in train_datasets:
        if dataset:
            return dataset[0]
    assert False, "No data, can't determine correct evaluation metrics"


if __name__ == '__main__':
    main()
