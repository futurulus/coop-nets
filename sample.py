import cPickle as pickle

import run_experiment  # NOQA: make sure we load all the command line args
import rsa_fit_data
from stanza.research import config

parser = config.get_options_parser()
parser.add_argument('--num_samples', type=int, default=100,
                    help='Number of samples to draw')


def output_sample(model):
    options = config.options()
    insts = model.sample_joint_smooth(num_samples=options.num_samples)
    html = rsa_fit_data.get_html(insts, title='Agent samples (smoothed prior)')
    config.dump([inst.__dict__ for inst in insts], 'data.sample.jsons', lines=True)
    with config.open('report.sample.html', 'w') as outfile:
        outfile.write(html)


if __name__ == '__main__':
    options = config.options(read=True)
    with config.open('model.p', 'r') as infile:
        output_sample(pickle.load(infile))
