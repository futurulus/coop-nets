import csv
import os
import StringIO

from stanza.research import config
from html_report import get_output


parser = config.get_options_parser()
parser.add_argument('--listener', type=config.boolean, default=False,
                    help='If True, create a listener "clickedObj" csv file. Otherwise '
                         'create a speaker "message" csv file.')

ID_COLUMNS = (0, 2)
SPEAKER_REPLACE_COLUMN = 4
COLOR_LOC = (8, 14, 20)
COLOR_BOUNDARY = (4, 10, 16, 22)


def generate_csv(run_dir=None):
    options = config.options(read=True)
    run_dir = run_dir or options.run_dir
    if options.listener:
        out_path = os.path.join(run_dir, 'clickedObj.csv')
        in_path = 'hawkins_data/colorReferenceClicks.csv'
    else:
        out_path = os.path.join(run_dir, 'message.csv')
        in_path = 'hawkins_data/colorReferenceMessage.csv'
    output = get_output(run_dir, 'eval')
    if 'error' in output.data[0]:
        output = get_output(run_dir, 'hawkins_dev')
    if 'error' in output.data[0]:
        output = get_output(run_dir, 'dev')

    with open(out_path, 'w') as outfile, open(in_path, 'r') as infile:
        outfile.write(csv_output(output, infile, listener=options.listener))


def csv_output(output, template_file, listener):
    buff = StringIO.StringIO()
    writer = csv.writer(buff)
    rows = [r for r in csv.reader(template_file)]
    row_table = build_row_table(rows)

    writer.writerow(rows[0])
    for inst_dict, pred in zip(output.data, output.predictions):
        orig_row = row_table[tuple(inst_dict['source'])]
        if listener:
            replaced_row = replace_row_listener(orig_row, pred)
        else:
            replaced_row = replace_row_speaker(orig_row, pred)
        writer.writerow(replaced_row)

    return buff.getvalue()


def build_row_table(rows):
    table = {}
    for row in rows:
        rowid = tuple(row[i] for i in ID_COLUMNS)
        table[rowid] = row
    return table


def replace_row_listener(orig_row, pred):
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

    return result


def replace_row_speaker(orig_row, pred):
    return orig_row[:-1] + [pred]


if __name__ == '__main__':
    generate_csv()
