from collections import OrderedDict
import cPickle as pickle
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
parser.add_argument('--aug_model', default=None,
                    help='Load a model from the given pickle file for use in generating '
                         'synthetic data. Should be a quickpickle file, to avoid '
                         'unexpected command line option interactions.')
parser.add_argument('--aug_noise_prob', type=float, default=0.0,
                    help='With this probability (fraction between 0 and 1), data '
                         'augmentation samples will be corrupted by randomizing the '
                         'target index. This is to prevent overconfidence, thereby '
                         'reducing perplexity.')


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

    - randomly changing the target with a configurable probability (--aug_noise_prob),
      without modifying the utterance. This helps prevent overconfidence.
    '''
    def sample_augmented(self, num_samples):
        self.get_sample_data()
        base_samples = self.sample_base(num_samples)
        return [self.mangle(s) for s in base_samples]

    def sample_base(self, num_samples):
        indices = rng.choice(np.arange(len(self.sample_data)), size=num_samples)
        return [self.sample_data[i] for i in indices]

    def mangle(self, inst):
        from fields import get_utt, get_color_index, get_context, build_instance

        list_input = self.options.listener
        list_output = self.is_listener
        utt = get_utt(inst, list_input)
        color_index = get_color_index(inst, list_input)
        context = get_context(inst, list_input)

        utt, color_index, context = self.negative_and_switch(utt, color_index, context)
        utt, color_index, context = self.target_noise(self.options.aug_noise_prob,
                                                      utt, color_index, context)
        utt, color_index, context = self.repeat(1, 3, utt, color_index, context)
        return build_instance(utt, color_index, context, list_output)

    def negative_and_switch(self, utt, color_index, context):
        if rng.choice([0, 1]):
            utt = 'not ' + utt
            others = list(range(0, color_index)) + list(range(color_index + 1, len(context)))
            color_index = rng.choice(others)
        return utt, color_index, context

    def target_noise(self, prob, utt, color_index, context):
        if rng.rand() <= prob:
            color_index = rng.choice(range(len(context)))
        return utt, color_index, context

    def repeat(self, min_repeat, max_repeat, utt, color_index, context):
        repeats = rng.choice(range(1, 4))
        utt = self.random_separators([utt] * repeats)
        return utt, color_index, context

    def random_separators(self, strings):
        separators = (' ', ', ', ' ~ ', ' <unk> ')
        chosen = [separators[i] for i in rng.choice(range(len(separators)),
                                                    size=len(strings) - 1)]
        return ''.join(string + sep for string, sep in zip(strings, chosen)) + strings[-1]

    def get_sample_data(self):
        self.get_options()
        if not hasattr(self, 'sample_data'):
            self.sample_data = (color_instances.SOURCES[self.options.aug_data_source]
                                               .train_data(listener=self.options.listener))


class SpeakerModelDataSampler(NotRepeatDataSampler):
    '''
    Data sampler that samples random instances from the source dataset, samples
    utterances for each color in the context of those instances from a speaker
    model (specified by --aug_model), and combines them with the modifications of
    NotRepeatDataSampler (adding "not" to the utterances that aren't the target).
    '''
    def __init__(self, id=None):
        super(SpeakerModelDataSampler, self).__init__(id=id)
        self.get_options()
        with open(self.options.aug_model, 'rb') as infile:
            self.speaker_model = pickle.load(infile)
        self.speaker_model.options.verbosity = 0

    def sample_augmented(self, num_samples):
        self.get_sample_data()
        base_samples = self.sample_base(num_samples)
        speaker_utts = self.get_context_speaker_utts(base_samples)
        return [self.mangle(s, utts) for s, utts in zip(base_samples, speaker_utts)]

    def get_context_speaker_utts(self, base_samples):
        from fields import get_context
        contexts = [get_context(inst, self.options.listener) for inst in base_samples]
        context_insts = [instance.Instance(c, None)
                         for context in contexts
                         for c in context]
        utts = self.speaker_model.predict(context_insts, random=True)
        grouped_utts = []
        utt_iter = iter(utts)
        for context in contexts:
            grouped_utts.append([next(utt_iter) for _ in range(len(context))])
        return grouped_utts

    def mangle(self, inst, speaker_utts):
        from fields import get_color_index, get_context, build_instance

        list_input = self.options.listener
        list_output = self.is_listener
        color_index = get_color_index(inst, list_input)
        context = get_context(inst, list_input)

        true_utt = speaker_utts[color_index]
        negated_utts = ['not ' + u for i, u in enumerate(speaker_utts) if i != color_index]

        if rng.rand() <= 0.95:
            chosen = [true_utt]
        else:
            i = rng.choice(range(len(negated_utts)))
            chosen = [negated_utts[i]]
            del negated_utts[i]

        if rng.rand() <= 0.15:
            for utt in negated_utts:
                if rng.rand() <= 0.5:
                    chosen.append(utt)

        rng.shuffle(chosen)
        utt = self.random_separators(chosen)

        utt, color_index, context = self.target_noise(self.options.aug_noise_prob,
                                                      utt, color_index, context)
        utt, color_index, context = self.repeat(1, 3, utt, color_index, context)
        return build_instance(utt, color_index, context, list_output)


AGENTS = {
    'NotRepeat': NotRepeatDataSampler,
    'ModelSampler': SpeakerModelDataSampler,
}
