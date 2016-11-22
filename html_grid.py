from numpy import argmax
from math import exp
import gzip
import json
from numbers import Number
import os

from stanza.research import config
from html_report import get_output, format_value


parser = config.get_options_parser()
parser.add_argument('--listener', type=config.boolean, default=False,
                    help='If True, create a listener "clickedObj" csv file. Otherwise '
                         'create a speaker "message" csv file.')
parser.add_argument('--suffix', type=str, default='',
                    help='Append this to the end of filenames (before the ".csv") when '
                         'locating the Hawkins data.')

ID_COLUMNS = (0, 2)
SPEAKER_REPLACE_COLUMN = 4
COLOR_LOC = (8, 14, 20)
COLOR_BOUNDARY = (4, 10, 16, 22)


def generate_html(run_dir=None):
    options = config.options(read=True)
    run_dir = run_dir or options.run_dir
    out_path = os.path.join(run_dir, 'grids.html')
    in_path = os.path.join(run_dir, 'grids.0.jsons.gz')
    output = get_output(run_dir, 'eval')
    if 'error' in output.data[0]:
        output = get_output(run_dir, 'hawkins_dev')
    if 'error' in output.data[0]:
        output = get_output(run_dir, 'dev')

    with open(out_path, 'w') as outfile, gzip.open(in_path, 'r') as infile:
        outfile.write(header(output))
        for example in read_grids(infile, output, options.only_differing_preds):
            outfile.write(grid_output(example, options.only_differing_preds))
        outfile.write(footer())


def header(output):
    return '''<!doctype html>
    <html>
    <head>
    <link rel="stylesheet" href="http://web.stanford.edu/~wmonroe4/css/style.css" type="text/css">
    <title>{run_dir} - Grids</title>
    </head>
    <body>
    '''.format(**output.config)


def read_grids(infile, output, only_differing_preds=True):
    show = []
    for inst_num, (inst, line) in enumerate(zip(output.data, infile)):
        grid = json.loads(line.strip())
        l2_pred = argmax(grid['final'])
        l0_grid = grid['sets'][0]['L0']
        l0_pred = argmax([l0_grid[i][0] for i in range(len(l0_grid))])
        if not only_differing_preds or l2_pred != l0_pred:
            show.append((inst_num, inst, grid))
    show.sort(key=prob_diff)
    for shown_num, (inst_num, inst, grid) in enumerate(show):
        yield (inst_num, shown_num, inst, grid)


def prob_diff(example):
    inst_num, inst, grids = example
    final_log_prob = grids['final'][inst['output']]
    l0_log_prob = grids['sets'][0]['L0'][inst['output']][0]
    return exp(l0_log_prob) - exp(final_log_prob)


def grid_output(example, only_differing_preds):
    inst_num, shown_num, inst, grids = example
    lines = [
        '<h3>Example {}{} {}</h3>'.format(
            inst_num + 1,
            ' [{}]'.format(shown_num + 1) if only_differing_preds else '',
            correct_status(inst, grids)
        ),
        '<table>',
        '<tr><td></td>{}</tr>'.format(colors_row(inst)),
        '<tr><td><b>{}</b></td>{}</tr>'.format(grids['sets'][0]['utts'][0],
                                               probs_row(grids['final'], bold=True))
    ]
    for i, ss in enumerate(grids['sets']):
        lines.append('<tr> <th></th>'
                     '<th>L2</th> <th></th> <th></th> <th></th>'
                     '<th>S1</th> <th></th> <th></th> <th></th>'
                     '<th>L0</th> </tr>')
        lines.append('<tr><td></td>{}<td>&emsp;</td>{}<td>&emsp;</td>{}</tr>'.format(*[
            colors_row(inst)
        ] * 3))
        for j, utt in enumerate(ss['utts']):
            lines.append('<tr><td>{utt}</td>{}<td>&emsp;</td>{}<td>&emsp;</td>{}</tr>'.format(*[
                probs_row([ss[model][k][j] for k in range(len(ss[model]))],
                          bold=model.startswith('L'))
                for model in ['L2', 'S1', 'L0']
            ], utt=utt))
    lines.append('</table>')
    return '\n'.join(lines)


def colors_row(inst):
    return ''.join([
        star_true(format_value(c, suppress_colors=True), i == inst['output'])
        for i, c in enumerate(inst['alt_outputs'])
    ])


def correct_status(inst, grids):
    l0_grid = grids['sets'][0]['L0']
    l0_pred = argmax([l0_grid[i][0] for i in range(len(l0_grid))])
    l0_correct = l0_pred == inst['output']
    l2_pred = argmax(grids['final'])
    l2_correct = l2_pred == inst['output']
    if l2_correct and l0_correct:
        return '(+)'
    elif l2_correct and not l0_correct:
        return '<font color="blue">(- &rightarrow; +)</font>'
    elif not l2_correct and l0_correct:
        return '<font color="red">(+ &rightarrow; -)</font>'
    elif l0_pred != l2_pred:
        return '<font color="olive">(- &rightarrow; -)</font>'
    else:
        return '(-)'


def star_true(formatted, is_true):
    formatted = formatted.replace('td', 'th')
    if is_true:
        return formatted.replace('&nbsp;', '*')
    else:
        return formatted


def probs_row(probs, bold=True):
    bold_idx = argmax(probs)
    return ''.join([
        format_prob(exp(p), bold=bold and (i == bold_idx))
        for i, p in enumerate(probs)
    ])


def format_prob(prob, bold=False):
    r = int(255.0 * (1.0 - prob))
    g = int(255.0 * (1.0 - 0.5 * prob))
    b = int(255.0 * (1.0 - prob))
    return '<td bgcolor="#{r:02x}{g:02x}{b:02x}">{prob_str}</td>'.format(
        r=r, g=g, b=b,
        prob_str=('<b>{}</b>' if bold else '{}').format(format_number(prob))
    )


def format_number(value):
    if not isinstance(value, Number):
        return repr(value)
    elif isinstance(value, int):
        return '{:,d}'.format(value)
    elif value > 1e8 or abs(value) < 1e-3:
        return '{:.0e}'.format(value)
    else:
        return '{:,.3f}'.format(value)


def footer():
    return '''</body>
    </html>'''


if __name__ == '__main__':
    generate_html()
