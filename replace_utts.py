import glob
import json
import os

from stanza.research import config
from html_report import get_output

parser = config.get_options_parser()
parser.add_argument('--speaker_dir', default=None,
                    help='A directory containing a predictions.*.jsons file giving '
                         'the values to insert in the "input" field of the data.')
parser.add_argument('--model_name', default='replaced',
                    help='An identifier to add to the name of the output file to '
                         'distinguish different data files replaced with the output '
                         'of different models.')


def output_replaced_data(run_dir=None):
    options = config.options(read=True)
    run_dir = run_dir or options.run_dir

    for output, preds, out_filename in get_all_outputs(run_dir, options.speaker_dir, options.model_name):
        config.dump(replaced_data(output, preds), out_filename, lines=True)


def get_all_outputs(run_dir, speaker_dir, model_name):
    for filename in glob.glob(os.path.join(run_dir, 'data.*.jsons')):
        split = os.path.basename(filename).split('.')[-2]
        this_output = get_output(run_dir, split)
        with open(os.path.join(speaker_dir, 'predictions.%s.jsons' % split)) as infile:
            predictions = [json.loads(line.strip()) for line in infile]

        out_filename = 'data_%s.%s.jsons' % (model_name, split)
        yield this_output, predictions, out_filename


def replaced_data(output, preds):
    result = output.data
    for inst_dict, pred in zip(result, preds):
        inst_dict['input'] = pred
    return result


if __name__ == '__main__':
    output_replaced_data()
