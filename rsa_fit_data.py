import sys
from collections import Counter

from stanza.research import config

import colordesc
from color_instances import get_training_instances
from speaker import UniformContextPrior
from ref_game import ExhaustiveS1Learner
from html_report import format_value


def generate_dataset(num_insts=100, print_html=True):
    cd = colordesc.ColorDescriber()
    get_prior_counter()
    exh = ExhaustiveS1Learner(cd.model)
    contexts = UniformContextPrior().sample(num_insts)
    context_size = len(contexts[0].alt_inputs)
    if print_html:
        print('<html><head><title>RSA data</title></head><body>')
        print('<table><tr><th>context</th>' + '<th></th>' * (context_size - 1) +
              '<th>target</th><th>prediction</th></tr>')
    insts = []
    for inst, pred in zip(contexts, exh.predict(contexts, random=True)):
        inst = inst.inverted()
        inst.input = pred
        insts.append(inst)
        cols = list(inst.alt_outputs) + [inst.output, pred]
        if print_html:
            print('<tr>')
            print(''.join(format_value(v) for v in cols))
            print('</tr>')
    if print_html:
        print('</table></body></html>')
    return insts


def get_prior_counter():
    sys.stderr.write('Counting utterances for prior...')
    sys.stderr.flush()
    return Counter(inst.output for inst in get_training_instances(listener=False))
    sys.stderr.write('done!\n')
    sys.stderr.flush()


if __name__ == '__main__':
    insts = generate_dataset(num_insts=100, print_html=False)
    config.dump([inst.__dict__ for inst in insts], 'data.out.jsons', lines=True)
