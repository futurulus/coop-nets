import numpy as np
from scipy.misc import logsumexp
from sklearn.linear_model import LogisticRegression

from stanza.monitoring import progress
from stanza.research import config, instance, iterators
from stanza.research.learner import Learner

from neural import sample
from tokenizers import TOKENIZERS
from vectorizers import SequenceVectorizer, COLOR_REPRS
import color_instances


class ExhaustiveS1Learner(Learner):
    def __init__(self, base=None):
        options = config.options()
        if base is None:
            self.base = learners.new(options.exhaustive_base_learner)
        else:
            self.base = base

    def train(self, training_instances, validation_instances=None, metrics=None):
        return self.base.train(training_instances=training_instances,
                               validation_instances=validation_instances, metrics=metrics)

    @property
    def num_params(self):
        return self.base.num_params

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        options = config.options()
        predictions = []
        scores = []

        all_utts = self.base.seq_vec.tokens
        sym_vec = vectorizers.SymbolVectorizer()
        sym_vec.add_all(all_utts)
        prior_scores = self.prior_scores(all_utts)

        base_is_listener = (type(self.base) in listener.LISTENERS.values())

        true_batch_size = options.listener_eval_batch_size / len(all_utts)
        batches = iterators.iter_batches(eval_instances, true_batch_size)
        num_batches = (len(eval_instances) - 1) // true_batch_size + 1

        if options.verbosity + verbosity >= 2:
            print('Testing')
        progress.start_task('Eval batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)
            context = len(batch[0].alt_inputs) if batch[0].alt_inputs is not None else 0
            if context:
                output_grid = [(instance.Instance(utt, color)
                                if base_is_listener else
                                instance.Instance(color, utt))
                               for inst in batch for color in inst.alt_inputs
                               for utt in sym_vec.tokens]
                assert len(output_grid) == context * len(batch) * len(all_utts), \
                    'Context must be the same number of colors for all examples'
                true_indices = np.array([inst.input for inst in batch])
            else:
                output_grid = [(instance.Instance(utt, inst.input)
                                if base_is_listener else
                                instance.Instance(inst.input, utt))
                               for inst in batch for utt in sym_vec.tokens]
                true_indices = sym_vec.vectorize_all([inst.input for inst in batch])
                if len(true_indices.shape) == 2:
                    # Sequence vectorizer; we're only using single tokens for now.
                    true_indices = true_indices[:, 0]
            scores = self.base.score(output_grid, verbosity=verbosity)
            if context:
                log_probs = np.array(scores).reshape((len(batch), context, len(all_utts)))
                orig_log_probs = log_probs[np.arange(len(batch)), true_indices, :]
                # Renormalize over only the context colors, and extract the score of
                # the true color.
                log_probs -= logsumexp(log_probs, axis=1)[:, np.newaxis, :]
                log_probs = log_probs[np.arange(len(batch)), true_indices, :]
            else:
                log_probs = np.array(scores).reshape((len(batch), len(all_utts)))
                orig_log_probs = log_probs
            assert log_probs.shape == (len(batch), len(all_utts))
            # Add in the prior scores, if used (S1 \propto L0 * P)
            if prior_scores is not None:
                log_probs = log_probs + 0.5 * prior_scores
            if options.exhaustive_base_weight:
                w = options.exhaustive_base_weight
                log_probs = w * orig_log_probs + (1.0 - w) * log_probs
            # Normalize across utterances. Note that the listener returns probability
            # densities over colors.
            log_probs -= logsumexp(log_probs, axis=1)[:, np.newaxis]
            if random:
                pred_indices = sample(np.exp(log_probs))
            else:
                pred_indices = np.argmax(log_probs, axis=1)
            predictions.extend(sym_vec.unvectorize_all(pred_indices))
            scores.extend(log_probs[np.arange(len(batch)), true_indices].tolist())
        progress.end_task()

        return predictions, scores

    def dump(self, outfile):
        return self.base.dump(outfile)

    def load(self, infile):
        return self.base.load(infile)

    def prior_scores(self, utts):
        # Don't use prior scores by default
        pass


class ExhaustiveS1PriorLearner(ExhaustiveS1Learner):
    def __init__(self, prior_counter, base=None):
        self.prior_counter = prior_counter
        self.denominator = sum(prior_counter.values())
        super(ExhaustiveS1PriorLearner, self).__init__(base=base)

    def prior_scores(self, utts):
        return np.log(np.array([self.prior_counter[u] for u in utts])) - np.log(self.denominator)


class ExhaustiveL2Learner(Learner):
    def __init__(self, base=None, sampler=None):
        options = self.get_options()
        if base is None:
            self.base = learners.new(options.exhaustive_base_learner)
        else:
            self.base = base
        if sampler is None and options.exhaustive_num_samples > 0:
            if options.verbosity >= 2:
                print('Loading sampler')
            self.sampler = learners.new(options.exhaustive_sampler_learner)
            with open(options.exhaustive_sampler_model, 'rb') as infile:
                self.sampler.load(infile)
        else:
            self.sampler = sampler

    def train(self, training_instances, validation_instances=None, metrics=None):
        raise NotImplementedError

    @property
    def num_params(self):
        # options = self.get_options()
        total = self.base.num_params
        # This is causing pickle problems; ignore these parameters for now
        # if options.exhaustive_num_samples > 0:
        #    total += self.sampler.num_params
        return total

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        options = self.get_options()
        predictions = []
        scores = []

        if options.verbosity + verbosity >= 2:
            print('Building alternative utterance list')
        sym_vec = vectorizers.SymbolVectorizer()
        sym_vec.add_all([inst.input for inst in self.get_dataset(self.base)])

        assert eval_instances[0].alt_outputs, \
            'Context required for L(S(L)): %s' % eval_instances[0].__dict__
        context_len = len(eval_instances[0].alt_outputs)
        if options.exhaustive_num_samples > 0:
            num_alt_utts = options.exhaustive_num_samples * context_len + 1
            num_sample_sets = options.exhaustive_num_sample_sets
        else:
            num_alt_utts = len(sym_vec.tokens) + 1
            num_sample_sets = 1
        true_batch_size = max(options.listener_eval_batch_size /
                              (num_alt_utts * num_sample_sets * context_len), 1)
        batches = iterators.iter_batches(eval_instances, true_batch_size)
        num_batches = (len(eval_instances) - 1) // true_batch_size + 1

        if options.verbosity + verbosity >= 2:
            print('Testing')
        progress.start_task('Eval batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)
            output_grid = self.build_grid(batch, sym_vec.tokens)
            assert len(output_grid) == len(batch) * num_sample_sets * context_len * num_alt_utts, \
                'Context must be the same number of colors for all examples %s' % \
                ((len(output_grid), len(batch), num_sample_sets, context_len, num_alt_utts),)
            true_indices = np.array([inst.output for inst in batch])
            grid_scores = self.base.score(output_grid, verbosity=verbosity)
            l0_log_probs = np.array(grid_scores).reshape((len(batch), num_sample_sets,
                                                          context_len, num_alt_utts))
            # Renormalize over only the context colors, and extract the score of
            # the true color according to the base model.
            l0_log_probs -= logsumexp(l0_log_probs, axis=2)[:, :, np.newaxis, :]
            assert l0_log_probs.shape == (len(batch), num_sample_sets,
                                          context_len, num_alt_utts), l0_log_probs.shape
            orig_log_probs = l0_log_probs[np.arange(len(batch)), 0, :, 0]
            assert orig_log_probs.shape == (len(batch), context_len), orig_log_probs.shape
            # Apply temperature parameter before speaker.
            utilities = options.exhaustive_inv_temperature * l0_log_probs
            # Normalize across utterances. Note that the listener returns probability
            # densities over colors.
            s1_log_probs = utilities - logsumexp(utilities, axis=3)[:, :, :, np.newaxis]
            assert s1_log_probs.shape == (len(batch), num_sample_sets,
                                          context_len, num_alt_utts), s1_log_probs.shape
            if options.exhaustive_output_speaker_samples or \
                    options.exhaustive_output_speaker_predictions:
                speaker_dist = s1_log_probs[np.arange(len(batch)), :, true_indices, :]
                if options.exhaustive_output_speaker_samples:
                    speaker_sample_indices = sample(np.exp(speaker_dist))
                    self.write_speaker_utterances('s0_samples.%s.jsons', output_grid,
                                                  speaker_sample_indices, l0_log_probs.shape)
                if options.exhaustive_output_speaker_predictions:
                    speaker_pred_indices = np.argmax(speaker_dist, axis=1)
                    self.write_speaker_utterances('s0_predictions.%s.jsons', output_grid,
                                                  speaker_pred_indices, l0_log_probs.shape)
            # Normalize again across context colors.
            l2_log_probs = s1_log_probs - logsumexp(s1_log_probs, axis=2)[:, :, np.newaxis, :]
            assert l2_log_probs.shape == (len(batch), num_sample_sets,
                                          context_len, num_alt_utts), l2_log_probs.shape
            # Extract the score of each color for the input utterance according to the L2 model.
            log_probs = l2_log_probs[np.arange(len(batch)), :, :, 0]
            assert log_probs.shape == (len(batch), num_sample_sets, context_len), log_probs.shape
            # Blend L0 and L2 (if enabled) to produce final score.
            if options.exhaustive_base_weight:
                w = options.exhaustive_base_weight
                log_probs = w * orig_log_probs[:, np.newaxis, :] + (1.0 - w) * log_probs
            # Normalize across context one more time to prevent cheating when
            # blending.
            log_probs -= logsumexp(log_probs, axis=2)[:, :, np.newaxis]
            # Average (in probability space) over sample sets
            log_probs = logsumexp(log_probs, axis=1) - np.log(log_probs.shape[1])
            if random:
                pred_indices = sample(np.exp(log_probs))
            else:
                pred_indices = np.argmax(log_probs, axis=1)
            predictions.extend(pred_indices)
            # Extract the score of the true color according to the combined model.
            scores.extend(log_probs[np.arange(len(batch)), true_indices].tolist())
        progress.end_task()

        return predictions, scores

    def write_speaker_utterances(self, file_pattern, output_grid, indices, tensor_shape):
        batch_size, num_sample_sets, context_len, num_alt_utts = tensor_shape
        for i in range(num_sample_sets):
            utts = []
            sample_set_indices = indices[:, i]
            for j, index in enumerate(sample_set_indices):
                utts.append(output_grid[np.ravel_multi_index((j, i, 0, index),
                                                             tensor_shape)].input)
            config.dump(utts, file_pattern % (i,), lines=True)

    def get_output_grid_index(self, tensor_shape, indices):
        stride = 1
        result = 0
        for dim, idx in reversed(zip(tensor_shape, indices)):
            result += idx * stride
            stride *= dim
        return result

    def build_grid(self, batch, all_utts):
        # for inst in batch:
        #     for j in range(num_sample_sets):
        #         for i in range(len(inst.context)):
        #             for utt in sample_utts(inst.context, i):
        #                 (utt -> inst.context, i)
        options = self.get_options()
        if options.exhaustive_num_samples > 0:
            sampler_inputs = [instance.Instance(i, None, alt_inputs=inst.alt_outputs)
                              for inst in batch
                              for _ in range(options.exhaustive_num_sample_sets)
                              for i in range(len(inst.alt_outputs))
                              for _ in range(options.exhaustive_num_samples)]
            context_len = len(batch[0].alt_outputs)
            assert len(sampler_inputs) == (len(batch) *
                                           options.exhaustive_num_sample_sets *
                                           context_len *
                                           options.exhaustive_num_samples), \
                'Building grid: inconsistent context length %s' % \
                (len(sampler_inputs), len(batch), options.exhaustive_num_sample_sets,
                 context_len, options.exhaustive_num_samples)
            outputs = self.sampler.sample(sampler_inputs)
            outputs = (np.array(outputs)
                         .reshape(len(batch), options.exhaustive_num_sample_sets,
                                  context_len * options.exhaustive_num_samples)
                         .tolist())

            return [instance.Instance(utt, j, alt_outputs=inst.alt_outputs)
                    for inst, sample_sets in zip(batch, outputs)
                    for samples in sample_sets
                    for j in range(len(inst.alt_outputs))
                    for utt in [inst.input] + samples]
        else:
            return [instance.Instance(utt, j, alt_outputs=inst.alt_outputs)
                    for inst in batch
                    for j in range(len(inst.alt_outputs))
                    for utt in [inst.input] + all_utts]

    def get_dataset(self, model):
        if hasattr(model, 'options'):
            options = model.options
        else:
            options = config.options()
        data_sources = options.data_source
        if not isinstance(data_sources, list):
            data_sources = [data_sources]
        train_sizes = options.train_size
        if not isinstance(train_sizes, list):
            train_sizes = [train_sizes]
        return [
            inst
            for data_source, train_size in zip(data_sources, train_sizes)
            for inst in color_instances.SOURCES[data_source].train_data(listener=True)[:train_size]
        ]

    def dump(self, outfile):
        return self.base.dump(outfile)

    def load(self, infile):
        return self.base.load(infile)

    def get_options(self):
        if not hasattr(self, 'options'):
            self.options = config.options()
        return self.options


class DirectRefGameLearner(Learner):
    def __init__(self, base=None):
        options = self.get_options()
        base_is_listener = self.override_listener(exists=False)
        old_listener = options.listener
        options.listener = base_is_listener
        if base is None:
            self.base = learners.new(options.direct_base_learner)
        else:
            self.base = base
        options.listener = old_listener

    def get_options(self):
        if not hasattr(self, 'options'):
            self.options = config.options()
        return self.options

    def train(self, training_instances, validation_instances=None, metrics=None):
        self.override_listener()
        return self.base.train(training_instances=training_instances,
                               validation_instances=validation_instances, metrics=metrics)

    def override_listener(self, exists=True):
        if self.options.direct_base_is_listener > 0:
            base_is_listener = True
            if exists:
                self.base.options.listener = True
        elif self.options.direct_base_is_listener < 0:
            base_is_listener = False
            if exists:
                self.base.options.listener = False
        else:
            base_is_listener = (self.options.direct_base_learner in listener.LISTENERS)
        return base_is_listener

    @property
    def num_params(self):
        return self.base.num_params

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        from fields import build_instance

        options = self.get_options()
        predictions = []
        scores = []
        base_is_listener = self.override_listener()
        assert options.listener, 'Eval data should be listener data for DirectRefGameLearner'

        true_batch_size = options.listener_eval_batch_size / options.num_distractors
        batches = iterators.iter_batches(eval_instances, true_batch_size)
        num_batches = (len(eval_instances) - 1) // true_batch_size + 1

        if options.verbosity + verbosity >= 2:
            print('Testing')
        progress.start_task('Eval batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)
            assert batch[0].alt_outputs, 'No context given for direct listener testing'
            context = len(batch[0].alt_outputs)
            if self.options.direct_base_uses_context:
                output_grid = [build_instance(inst.input, target, inst.alt_outputs,
                                              base_is_listener)
                               for inst in batch for target in range(len(inst.alt_outputs))]
            else:
                output_grid = [build_instance(inst.input, color, None, base_is_listener)
                               for inst in batch for color in inst.alt_outputs]
            assert len(output_grid) == context * len(batch), \
                'Context must be the same number of colors for all examples'
            true_indices = np.array([inst.output for inst in batch])
            grid_scores = self.base.score(output_grid, verbosity=verbosity)
            log_probs = np.array(grid_scores).reshape((len(batch), context))
            # Renormalize over only the context colors
            log_probs -= logsumexp(log_probs, axis=1)[:, np.newaxis]
            # Cap confidences to reasonable values
            if options.direct_min_score is not None and options.direct_min_score <= 0.0:
                log_probs = np.maximum(options.direct_min_score, log_probs)
                # Normalize again (so we always return log probabilities)
                log_probs -= logsumexp(log_probs, axis=1)[:, np.newaxis]
            assert log_probs.shape == (len(batch), context)
            pred_indices = np.argmax(log_probs, axis=1)
            predictions.extend(pred_indices.tolist())
            # Extract the score of the true color
            scores.extend(log_probs[np.arange(len(batch)), true_indices].tolist())
        progress.end_task()

        return predictions, scores

    def dump(self, outfile):
        return self.base.dump(outfile)

    def load(self, infile):
        options = self.get_options()
        base_is_listener = self.override_listener(exists=False)
        old_listener = options.listener
        options.listener = base_is_listener
        result = self.base.load(infile)
        options.listener = old_listener
        return result


class LRContextListenerLearner(Learner):
    def train(self, training_instances, validation_instances=None, metrics=None):
        X, y = self._data_to_arrays(training_instances, init_vectorizer=True)
        self.mod = LogisticRegression(solver='lbfgs')
        self.mod.fit(X, y)

    @property
    def num_params(self):
        return np.prod(self.mod.coef_.shape) + np.prod(self.mod.intercept_.shape)

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        X, y = self._data_to_arrays(eval_instances)
        y = y.reshape((len(eval_instances), self.context_len))
        all_scores = self.mod.predict_log_proba(X)[:, 1].reshape((len(eval_instances),
                                                                  self.context_len))
        all_scores -= logsumexp(all_scores, axis=1)[:, np.newaxis]

        preds = all_scores.argmax(axis=1)
        scores = np.where(y, all_scores, 0).sum(axis=1)

        return preds.tolist(), scores.tolist()

    def _data_to_arrays(self, instances, inverted=False, init_vectorizer=False):
        self.get_options()

        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_desc, get_color = (get_o, get_i) if inverted else (get_i, get_o)
        get_alt_i, get_alt_o = (lambda inst: inst.alt_inputs), (lambda inst: inst.alt_outputs)
        get_alt_colors = get_alt_i if inverted else get_alt_o

        tokenize = TOKENIZERS[self.options.listener_tokenizer]
        tokenized = [tokenize(get_desc(inst)) for inst in instances]
        context_lens = [len(get_alt_colors(inst)) for inst in instances]

        if init_vectorizer:
            self.seq_vec = SequenceVectorizer()
            self.seq_vec.add_all(tokenized)

        unk_replaced = self.seq_vec.unk_replace_all(tokenized)

        if init_vectorizer:
            config.dump(unk_replaced, 'unk_replaced.train.jsons', lines=True)

            self.context_len = context_lens[0]

            color_repr = COLOR_REPRS[self.options.listener_color_repr]
            self.color_vec = color_repr(self.options.listener_color_resolution,
                                        hsv=self.options.listener_hsv)

        assert all(cl == self.context_len for cl in context_lens), (self.context_len, context_lens)

        padded = [(d + ['</s>'] * (self.seq_vec.max_len - len(d)))[:self.seq_vec.max_len]
                  for d in unk_replaced]
        colors = [c for inst in instances for c in get_alt_colors(inst)]
        labels = np.array([int(i == get_color(inst))
                           for inst in instances
                           for i in range(self.context_len)])

        desc_indices = self.seq_vec.vectorize_all(padded)
        desc_bow = -np.ones((desc_indices.shape[0], self.seq_vec.num_types))
        desc_bow[np.arange(desc_indices.shape[0])[:, np.newaxis], desc_indices] = 1.
        color_feats = self.color_vec.vectorize_all(colors)
        color_feats = color_feats.reshape((desc_indices.shape[0],
                                           self.context_len,
                                           color_feats.shape[1]))
        feats = np.einsum('ij,ick->icjk', desc_bow, color_feats)
        feats = feats.reshape((desc_indices.shape[0] * self.context_len,
                               desc_bow.shape[1] * color_feats.shape[2]))

        return feats, labels

    def get_options(self):
        if not hasattr(self, 'options'):
            self.options = config.options()


import learners
import listener
import vectorizers


parser = config.get_options_parser()
parser.add_argument('--exhaustive_base_learner', default='Listener',
                    choices=learners.LEARNERS.keys(),
                    help='The name of the model to use as the L0 for exhaustive enumeration-based '
                         'models.')
parser.add_argument('--exhaustive_base_weight', default=0.0, type=float,
                    help='Weight given to the base agent for the exhaustive RSA model. The RSA '
                         "agent's weight will be 1 - exhaustive_base_weight.")
parser.add_argument('--exhaustive_inv_temperature', default=1.0, type=float,
                    help="RSA inverse temperature parameter (lambda/alpha) for "
                         "ExhaustiveL2Learner. (Not yet implemented in ExhaustiveS1Learner as of "
                         "9/25/2016.)")
parser.add_argument('--exhaustive_sampler_learner', default='Speaker',
                    choices=learners.LEARNERS.keys(),
                    help='The name of the model to use as the speaker for sampling utterances in '
                         'exhaustive enumeration-based models.')
parser.add_argument('--exhaustive_sampler_model', default=None,
                    help='The path to the model to use as the speaker for sampling utterances in '
                         'exhaustive enumeration-based models.')
parser.add_argument('--exhaustive_num_samples', default=0, type=int,
                    help='The number of samples to take per context color for use as alternative '
                         'utterances. If 0 or negative, use the entire training corpus.')
parser.add_argument('--exhaustive_num_sample_sets', default=1, type=int,
                    help='The number of sets of alternative utterances to sample. L2 probabilities '
                         'will be averaged over alternative sets. Should be at least 1. Not used '
                         'if exhaustive_num_samples <= 0.')
parser.add_argument('--exhaustive_output_speaker_samples', default=False, type=config.boolean,
                    help='If True, write a file to the run directory containing an utterance '
                         'sampled from S(L0) [only for ExhaustiveL2Learner] for each test '
                         'instance.')
parser.add_argument('--exhaustive_output_speaker_predictions', default=False, type=config.boolean,
                    help='If True, write a file to the run directory containing the top-1 '
                         'utterance from S(L0) [only for ExhaustiveL2Learner] for each test '
                         'instance.')
parser.add_argument('--direct_base_learner', default='Listener',
                    choices=learners.LEARNERS.keys(),
                    help='The name of the model to use as the level-0 agent for direct score-based '
                         'listener models.')
parser.add_argument('--direct_base_is_listener', default=0, type=int,
                    help='If +1, override the --listener option in the base learner so it '
                         'becomes a listener. If -1, override so it becomes a speaker. If 0, '
                         'use the --listener option. Only useful for learners that can be '
                         'either speakers or listeners (e.g. RSA).')
parser.add_argument('--direct_base_uses_context', default=False, type=config.boolean,
                    help='If True, pass context and a target index through to the base learner. '
                         'Otherwise, extract the target color itself and discard remaining '
                         'context.')
parser.add_argument('--direct_min_score', default=None, type=float,
                    help='The log likelihood of the base model will be capped from below to this '
                         'value. This prevents extreme-confidence wrong decisions, and '
                         'is roughly equivalent to postulating an a priori probability that a '
                         'target in the dataset is chosen uniformly at random. None or positive '
                         'values will be interpreted as no cap.')
