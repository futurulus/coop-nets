from numpy import argmax, isfinite
from math import exp
import gzip
import itertools
import json
import numpy as np
from numbers import Number
import os
from xml.sax.saxutils import escape as html_escape

from stanza.research import config
from html_report import get_output, format_value
from blending import normalize, log_average


parser = config.get_options_parser()
parser.add_argument('--listener', type=config.boolean, default=False,
                    help='If True, create a listener "clickedObj" csv file. Otherwise '
                         'create a speaker "message" csv file.')
parser.add_argument('--limit_sample_sets', type=int, default=0,
                    help='If positive, show at most this number of sample sets.')
parser.add_argument('--baseline', choices=['l0', 'l1', 'l2', 'la', 'lb', 'le'], default='l0',
                    help='The listener model to use as the baseline when reporting '
                         'improved/declined (+/-).')
parser.add_argument('--compare', choices=['l0', 'l1', 'l2', 'la', 'lb', 'le'], default='le',
                    help='The listener model to use as the candidate when reporting '
                         'improved/declined (+/-).')

ID_COLUMNS = (0, 2)
SPEAKER_REPLACE_COLUMN = 4
COLOR_LOC = (8, 14, 20)
COLOR_BOUNDARY = (4, 10, 16, 22)


def generate_html(run_dir=None):
    options = config.options(read=True)
    run_dir = run_dir or options.run_dir
    out_path = os.path.join(run_dir, 'grids.html')
    try:
        in_path = os.path.join(run_dir, 's0_grids.0.jsons.gz')
        with open(in_path, 'r'):
            pass
    except IOError:
        in_path = os.path.join(run_dir, 'grids.0.jsons.gz')
    output = get_output(run_dir, 'eval')
    if 'error' in output.data[0]:
        output = get_output(run_dir, 'hawkins_dev')
    if 'error' in output.data[0]:
        output = get_output(run_dir, 'dev')

    with open(out_path, 'w') as outfile, gzip.open(in_path, 'r') as infile:
        write_files(infile, outfile, output, options)


def write_files(infile, outfile, output, options):
        outfile.write(header(output))
        grids = read_grids(infile, output, options)
        for example in grids:
            outfile.write(grid_output(example, options.only_differing_preds,
                                      options.limit_sample_sets))
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


def read_grids(infile, output, options):
    show = []
    for inst_num, (inst, line) in enumerate(zip(output.data, infile)):
        grid = json.loads(line.strip())
        lb_pred = argmax(grid['final'])
        l0 = np.array([np.array(ss['L0']).T for ss in grid['sets']])
        l0_pred = l0[0, 0, :].argmax()
        l2 = np.array([np.array(ss['L2']).T for ss in grid['sets']])
        l2_pred = l2[0, 0, :].argmax()
        preds = {'lb': lb_pred, 'l2': l2_pred, 'l0': l0_pred}
        if 'S0' in grid['sets'][0]:
            s0 = np.array([np.array(ss['S0']).T for ss in grid['sets']])
            l1 = normalize(s0[0, 0, :], axis=0)
            assert l1.shape == (3,), l1.shape
            grid['L1'] = l1
            preds['l1'] = l1.argmax()

            sw = 0.608
            bw = -0.15
            alpha = 0.544
            gamma = 0.509

            la = normalize((1 - sw) * l0[0, 0, :] + sw * s0[0, 0, :], axis=0)
            assert la.shape == (3,), la.shape
            grid['La'] = la
            preds['la'] = la.argmax()

            s1 = normalize(l0 * alpha, axis=1)
            l2 = normalize(s1, axis=2)
            lb_ss = normalize(bw * l0[:, 0, :] + (1 - bw) * l2[:, 0, :], axis=1)
            lb = normalize(log_average(lb_ss, axis=0), axis=0)
            assert lb.shape == (3,), lb.shape
            grid['Lb'] = lb

            le = normalize((1 - gamma) * la + gamma * lb, axis=0)
            assert le.shape == (3,), le.shape
            grid['Le'] = le
            preds['le'] = le.argmax()

        if not options.only_differing_preds or preds[options.compare] != preds[options.baseline]:
            show.append((inst_num, inst, grid))
    show.sort(key=prob_diff(options))
    for shown_num, (inst_num, inst, grid) in enumerate(show):
        yield (inst_num, shown_num, inst, grid)


def prob_diff(options):
    def comparator(example):
        inst_num, inst, grids = example
        baseline_log_prob = get_log_prob(grids, inst, options.baseline)
        compare_log_prob = get_log_prob(grids, inst, options.compare)
        return exp(baseline_log_prob) - exp(compare_log_prob)

    return comparator


def get_log_prob(grids, inst, model_name):
    if model_name == 'l0':
        return grids['sets'][0]['L0'][inst['output']][0]
    elif model_name == 'l2':
        return grids['sets'][0]['L2'][inst['output']][0]
    elif model_name == 'la':
        return grids['La'][inst['output']]
    elif model_name == 'lb':
        return grids['Lb'][inst['output']]
    elif model_name == 'le':
        return grids['Le'][inst['output']]
    elif model_name == 'final':
        return grids['final'][inst['output']]
    else:
        raise ValueError('unknown model: {}'.format(model_name))


def grid_output(example, only_differing_preds, limit_sample_sets):
    if limit_sample_sets <= 0:
        limit_sample_sets = None
    inst_num, shown_num, inst, grids = example
    lines = [
        '<h3>Example {}{} {}</h3>'.format(
            inst_num + 1,
            ' [{}]'.format(shown_num + 1) if only_differing_preds else '',
            correct_status(inst, grids)
        ),
        '<table>',
        '<tr><td>Le</td>{}</tr>'.format(colors_row(inst)),
        '<tr><td><b>{}</b></td>{}</tr>'.format(escape(grids['sets'][0]['utts'][0]),
                                               probs_row(grids['Le'], bold=True)),
        '<tr><td>La</td>{}</tr>'.format(colors_row(inst)),
        '<tr><td><b>{}</b></td>{}</tr>'.format(escape(grids['sets'][0]['utts'][0]),
                                               probs_row(grids['La'], bold=True)),
        '<tr><td>Lb</td>{}</tr>'.format(colors_row(inst)),
        '<tr><td><b>{}</b></td>{}</tr>'.format(escape(grids['sets'][0]['utts'][0]),
                                               probs_row(grids['Lb'], bold=True)),
        '<tr><td>L1</td>{}</tr>'.format(colors_row(inst)),
        '<tr><td><b>{}</b></td>{}</tr>'.format(escape(grids['sets'][0]['utts'][0]),
                                               probs_row(grids['L1'], bold=True)),
    ]
    for i, ss in enumerate(grids['sets'][:limit_sample_sets]):
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
            ], utt=escape(utt)))
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


def escape(s):
    return html_escape(s).encode('utf-8')


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
    if isfinite(prob):
        r = int(255.0 * (1.0 - prob))
        g = int(255.0 * (1.0 - 0.5 * prob))
        b = int(255.0 * (1.0 - prob))
    else:
        r = 255
        g = 128
        b = 0
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
