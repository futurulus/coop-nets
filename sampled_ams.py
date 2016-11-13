from lasagne.layers import NonlinearityLayer, SliceLayer
from lasagne.nonlinearities import softmax
import numpy as np

from stanza.research import config, instance

import listener
from neural import SimpleLasagneModel
from helpers import logit_softmax_nd


class ACGaussianLearner(listener.GaussianContextListenerLearner):
    def __init__(self, sampler=None, *args, **kwargs):
        super(ACGaussianLearner, self).__init__(*args, **kwargs)
        self.get_options()
        if sampler is None:
            if self.options.verbosity >= 2:
                print('Loading sampler')
            self.sampler = learners.new(self.options.ac_sampler_learner)
            with open(self.options.ac_sampler_model, 'rb') as infile:
                self.sampler.load(infile)
        else:
            self.sampler = sampler

    def _build_model(self, model_class=SimpleLasagneModel):
        self.get_options()
        multi_utt = 1 + self.context_len * self.options.ac_num_samples
        return super(ACGaussianLearner, self)._build_model(model_class=model_class,
                                                           multi_utt=multi_utt)

    def _get_l_out(self, input_vars, multi_utt=None):
        id_tag = (self.id + '/') if self.id else ''
        l_l0, input_layers = super(ACGaussianLearner, self)._get_l_out(input_vars,
                                                                       multi_utt=multi_utt)
        l_s1 = NonlinearityLayer(l_l0, nonlinearity=logit_softmax_nd(axis=1),
                                 name=id_tag + 'log_s1')
        l_s1_true = SliceLayer(l_s1, 0, axis=1, name=id_tag + 'log_s1_true')
        l_l2 = NonlinearityLayer(l_s1_true, nonlinearity=softmax,
                                 name=id_tag + 'l2')
        return l_l2, input_layers

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        (_, c), (y,) = super(ACGaussianLearner,
                             self)._data_to_arrays(training_instances,
                                                   init_vectorizer=init_vectorizer,
                                                   test=test, inverted=inverted)

        self.get_options()
        (xs, _), (_,) = super(ACGaussianLearner,
                              self)._data_to_arrays(self.build_grid(training_instances),
                                                    init_vectorizer=False,
                                                    inverted=False)
        multi_utt = 1 + self.context_len * self.options.ac_num_samples
        xs = xs.reshape((len(training_instances), multi_utt, self.seq_vec.max_len))
        return [xs, c], [y]

    def build_grid(self, batch):
        options = self.options
        # for inst in batch:
        #     for i in range(len(inst.context)):
        #         for utt in sample_utts(inst.context, i):
        #             (utt -> inst.context, i)
        sampler_inputs = [instance.Instance(i, None, alt_inputs=inst.alt_outputs)
                          for inst in batch
                          for i in range(len(inst.alt_outputs))
                          for _ in range(options.ac_num_samples)]
        context_len = len(batch[0].alt_outputs)
        assert len(sampler_inputs) == (len(batch) *
                                       context_len *
                                       options.ac_num_samples), \
            'Building grid: inconsistent context length %s' % \
            (len(sampler_inputs), len(batch),
             context_len, options.ac_num_samples)
        outputs = self.sampler.sample(sampler_inputs)
        outputs = (np.array(outputs)
                     .reshape(len(batch),
                              context_len * options.ac_num_samples)
                     .tolist())

        return [instance.Instance(utt, 0, alt_outputs=[(0, 0, 0)] * 3)
                for inst, samples in zip(batch, outputs)
                for utt in [inst.input] + samples]


import learners


parser = config.get_options_parser()
parser.add_argument('--ac_sampler_learner', default='Speaker',
                    choices=learners.SPEAKERS.keys(),
                    help='The class of speaker model to use for obtaining alterative utterances '
                         'in the sampled Amsterdam Colloquium model.')
parser.add_argument('--ac_sampler_model', default=None,
                    help='The path to the speaker model to use for obtaining alterative utterances '
                         'in the sampled Amsterdam Colloquium model.')
parser.add_argument('--ac_num_samples', default=1, type=int,
                    help='The number of samples to take per context color for use as alternative '
                         'utterances.')
