# Usage:
#   python export_csv.py --run_dir runs/listener_filtered --listener true --filtered true --source literal
#   python export_csv.py --run_dir runs/speaker_filtered --filtered true --source literal
#   python export_csv.py --run_dir runs/listener_big --listener true --suffix 2
#   python export_csv.py --run_dir runs/speaker_big --suffix 2
# For exporting s1 samples:
#   mkdir runs/s1_samples
#   cp runs/l2/s1_samples.0.jsons runs/s1_samples/predictions.eval.jsons
#   cp runs/l2/data.eval.jsons runs/s1_samples/
#   echo '{}' > runs/s1_samples/config.json
#   touch runs/s1_samples/scores.eval.jsons
#   python export_csv.py --run_dir runs/s1_samples --filtered true --source pragmatic
import csv
import os
import StringIO
import warnings

from stanza.research import config
from html_report import get_output


parser = config.get_options_parser()
parser.add_argument('--listener', type=config.boolean, default=False,
                    help='If True, create a listener "clickedObj" csv file. Otherwise '
                         'create a speaker "message" csv file.')
parser.add_argument('--suffix', type=str, default='',
                    help='Append this to the end of filenames (before the ".csv") when '
                         'locating the Hawkins data.')
parser.add_argument('--filtered', type=config.boolean, default=False,
                    help='If True, look for the filteredCorpus csv file. --suffix should '
                         'be empty.')
parser.add_argument('--source', type=str, default='model',
                    help='"Source" entry for filtered csv (as opposed to "human").')

ID_COLUMNS = (0, 2)
SPEAKER_MESSAGE_COLUMN = 4
FILTERED_MESSAGE_COLUMN = 4
COLOR_LOC = (8, 14, 20)
COLOR_BOUNDARY = (4, 10, 16, 22)


def generate_csv(run_dir=None):
    options = config.options(read=True)
    run_dir = run_dir or options.run_dir
    in_path = 'behavioralAnalysis/humanOutput/filteredCorpus.csv' if options.filtered else None
    if options.listener:
        out_path = os.path.join(run_dir, 'clickedObj.csv')
        if not in_path:
            in_path = 'hawkins_data/colorReferenceClicks%s.csv' % options.suffix
    else:
        out_path = os.path.join(run_dir, 'message.csv')
        if not in_path:
            in_path = 'hawkins_data/colorReferenceMessage%s.csv' % options.suffix
    output = get_output(run_dir, 'eval')
    if 'error' in output.data[0]:
        output = get_output(run_dir, 'hawkins_dev')
    if 'error' in output.data[0]:
        output = get_output(run_dir, 'dev')

    with open(out_path, 'w') as outfile, open(in_path, 'r') as infile:
        outfile.write(csv_output(output, infile, listener=options.listener,
                                 source=options.source, filtered=options.filtered))


def csv_output(output, template_file, listener, source, filtered):
    buff = StringIO.StringIO()
    writer = csv.writer(buff, quoting=csv.QUOTE_ALL)
    rows = [r for r in csv.reader(template_file)]
    row_table = build_row_table(rows[1:])

    missing = 0
    written = 0
    writer.writerow(rows[0])
    for inst_dict, pred in zip(output.data, output.predictions):
        lookup_id = tuple(inst_dict['source'])[:len(ID_COLUMNS)]
        try:
            orig_row = row_table[lookup_id]
        except KeyError:
            warnings.warn('Missing row: %s' % (lookup_id,))
            missing += 1
            continue
        if listener:
            replaced_row = replace_row_listener(orig_row, pred, source)
        elif filtered:
            replaced_row = replace_row_speaker_filtered(orig_row, pred, source)
        else:
            replaced_row = replace_row_speaker(orig_row, pred)
        writer.writerow(replaced_row)
        written += 1

    print('%s written rows' % written)
    print('%s missing rows' % missing)

    return buff.getvalue()


def build_row_table(rows):
    table = {}
    for row in rows:
        rowid = tuple(row[i] for i in ID_COLUMNS)
        table[rowid] = row
    return table


def replace_row_listener(orig_row, pred, source):
    result = orig_row[:4]

    chunks = [orig_row[COLOR_BOUNDARY[i]:COLOR_BOUNDARY[i + 1]] for i in range(3)]
    chunks.sort(key=lambda c: int(c[4]))  # Discard human click data in favor of LocS
    click = chunks[pred]
    alt1, alt2 = [chunks[i] for i in range(3) if i != pred]
    result.extend(click)
    result.extend(alt1)
    result.extend(alt2)

    result.extend(orig_row[22:25])

    outcome = 'true' if click[0] == 'target' else 'false'
    result.append(outcome)

    result.extend(orig_row[26:])
    if len(result) > 26:
        assert result[-1] == 'human', result[-1]
        result[-1] = source

    return result


def replace_row_speaker(orig_row, pred):
    return orig_row[:SPEAKER_MESSAGE_COLUMN - 1] + ['speaker', pred.replace('"', '""')]


def replace_row_speaker_filtered(orig_row, pred, source):
    return (orig_row[:FILTERED_MESSAGE_COLUMN - 1] + ['speaker', pred.replace('"', '""')] +
            orig_row[FILTERED_MESSAGE_COLUMN + 1:-1] + [source])


if __name__ == '__main__':
    generate_csv()
