import sys
from collections import Counter

import colordesc
from color_instances import get_training_instances
from speaker import UniformContextPrior
from ref_game import ExhaustiveS1Learner
from html_report import format_value


def generate_dataset(num_insts=100):
    cd = colordesc.ColorDescriber()
    get_prior_counter()
    exh = ExhaustiveS1Learner(cd.model)
    insts = UniformContextPrior().sample(num_insts)
    context_size = len(insts[0].alt_inputs)
    print('<html><head><title>RSA data</title></head><body>')
    print('<table><tr><th>context</th>' + '<th></th>' * (context_size - 1) +
          '<th>target</th><th>prediction</th></tr>')
    for inst, pred in zip(insts, exh.predict(insts, random=True)):
        cols = list(inst.alt_inputs) + [inst.input, pred]
        print('<tr>')
        print(''.join(format_value(v) for v in cols))
        print('</tr>')
    print('</table></body></html>')


def get_prior_counter():
    sys.stderr.write('Counting utterances for prior...')
    sys.stderr.flush()
    return Counter(inst.output for inst in get_training_instances(listener=False))
    sys.stderr.write('done!\n')
    sys.stderr.flush()


if __name__ == '__main__':
    generate_dataset()
