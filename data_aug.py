from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer

from stanza.research import config, instance
from stanza.research.rng import get_rng

from neural import SimpleLasagneModel, NeuralLearner
import color_instances


rng = get_rng()

parser = config.get_options_parser()
parser.add_argument('--aug_data_source', default='hawkins_dev',
                    choices=color_instances.SOURCES.keys(),
                    help='The type of data to use.')


class DataSampler(NeuralLearner):
    '''
    Base class for dummy agents that only exist to produce sampled data.
    Training this agent is a no-op, and it can't make predictions, but its
    sampling methods work and can be used in an RSA coop-nets setup.

    Subclasses should override the sample_augmented method.
    '''
    def __init__(self, id=None):
        self.id = id
        self.seq_vec = None
        self.color_vec = None

    @property
    def is_listener(self):
        self.get_options()
        if self.id:
            id_comps = self.id.split('/')
            for comp in reversed(id_comps):
                if comp.startswith('L'):
                    return True
            return False
        else:
            return self.options.listener

    def train(self, training_instances, validation_instances=None, metrics=None):
        pass

    def get_options(self):
        if not hasattr(self, 'options'):
            self.options = config.options()

    def train_priors(self, training_instances, listener_data=False):
        pass

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        return ([np.zeros((len(training_instances),), dtype=theano.config.floatX)],
                [np.zeros((len(training_instances),), dtype=theano.config.floatX)])

    def _build_model(self, model_class=SimpleLasagneModel):
        id_tag = (self.id + '/') if self.id else ''

        input_var = T.vector(id_tag + 'dummy_in')
        target_var = T.vector(id_tag + 'dummy_out')

        self.l_out, self.input_layers = self._get_l_out([input_var])
        self.loss = add_loss  # 0.0 [input] + 0.0 [target] = 0.0 [loss]
        self.model = model_class([input_var], [target_var], self.l_out,
                                 loss=self.loss, optimizer=null_optimizer, id=self.id)

    def _get_l_out(self, input_vars):
        id_tag = (self.id + '/') if self.id else ''

        l_out = InputLayer(shape=(None,), input_var=input_vars[0],
                           name=id_tag + 'dummy_layer')
        return l_out, [l_out]

    def sample_joint_emp(self, num_samples):
        return self.sample_augmented(num_samples)

    def sample_joint_smooth(self, num_samples):
        return self.sample_augmented(num_samples)

    def sample_augmented(self, num_samples):
        raise NotImplementedError


def add_loss(predictions, targets):
    return predictions + targets


def null_optimizer(loss='ignored', params='ignored', learning_rate='ignored'):
    return OrderedDict()


class NotRepeatDataSampler(DataSampler):
    '''
    Data sampler that samples random instances from the training dataset and
    modifies them by:

    - randomly adding "not" to the beginning of the utterance with prob. 50%
      (and changing the target to another randomly-chosen distractor when this happens)

    - repeating the utterance (possibly with the "not") 1-3 times, separated by
      commas, newlines, or only a space
    '''
    def sample_augmented(self, num_samples):
        self.get_sample_data()
        indices = rng.choice(np.arange(len(self.sample_data)), size=num_samples)
        base_samples = [self.sample_data[i] for i in indices]
        return [self.mangle(s) for s in base_samples]

    def mangle(self, inst):
        get_input = lambda i: i.input
        get_output = lambda i: i.output
        get_alt_inputs = lambda i: i.alt_inputs
        get_alt_outputs = lambda i: i.alt_outputs
        get_utt = get_input if self.options.listener else get_output
        get_color_index = get_output if self.options.listener else get_input
        get_context = get_alt_outputs if self.options.listener else get_alt_inputs

        utt = get_utt(inst)
        color_index = get_color_index(inst)
        context = get_context(inst)

        if rng.choice([0, 1]):
            utt = 'not ' + utt
            others = list(range(0, color_index)) + list(range(color_index + 1, len(context)))
            color_index = rng.choice(others)

        repeats = rng.choice(range(1, 4))
        separators = [(' ', ', ', ' ~ ')[i] for i in rng.choice(range(3), size=repeats - 1)]
        utt += ''.join(s + utt for s in separators)

        if self.is_listener:
            return instance.Instance(utt, color_index, alt_outputs=context)
        else:
            return instance.Instance(color_index, utt, alt_inputs=context)

    def get_sample_data(self):
        self.get_options()
        if not hasattr(self, 'sample_data'):
            self.sample_data = (color_instances.SOURCES[self.options.aug_data_source]
                                               .train_data(listener=self.options.listener))


AGENTS = {
    'NotRepeat': NotRepeatDataSampler,
}
