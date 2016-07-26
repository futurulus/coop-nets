import sys
from collections import Counter

from stanza.research import config

import colordesc
from color_instances import get_training_instances
from speaker import UniformContextPrior
from ref_game import ExhaustiveS1Learner
from html_report import format_value


def get_html(insts, title='RSA data'):
    context_size = len(insts[0].alt_outputs)
    html = ('<html><head><title>%s</title></head><body>\n' % title +
            '<table><tr><th>context</th>' + '<th></th>' * (context_size - 1) +
            '<th>target</th><th>utterance</th></tr>\n')

    for inst in insts:
        cols = list(inst.alt_outputs) + [inst.output, inst.input]
        html += '<tr>\n%s\n</tr>\n' % (''.join(format_value(v) for v in cols))
    html += '</table></body></html>'

    return html


def generate_dataset(num_insts=100, print_html=True):
    cd = colordesc.ColorDescriber()
    get_prior_counter()  # load the data to keep random numbers the same
    exh = ExhaustiveS1Learner(cd.model)
    contexts = UniformContextPrior().sample(num_insts)
    insts = []
    for inst, pred in zip(contexts, exh.predict(contexts, random=True)):
        inst = inst.inverted()
        inst.input = pred
        insts.append(inst)
    if print_html:
        print(get_html(insts))
    return insts


def get_prior_counter():
    sys.stderr.write('Counting utterances for prior...')
    sys.stderr.flush()
    c = Counter(inst.output for inst in get_training_instances(listener=False))
    sys.stderr.write('done!\n')
    sys.stderr.flush()
    return c


if __name__ == '__main__':
    insts = generate_dataset(num_insts=100, print_html=False)
    config.dump([inst.__dict__ for inst in insts], 'data.out.jsons', lines=True)
