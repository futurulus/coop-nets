# -*- coding: utf-8 -*-

import listener
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, EmbeddingLayer, NINLayer, MergeLayer, dimshuffle
from lasagne.init import Normal
import numpy as np

from stanza.research import config

from neural import SimpleLasagneModel
from vectorizers import SymbolVectorizer

parser = config.get_options_parser()
parser.add_argument('--bilingual_embed_size', default=100,
                    help='Size of word embeddings to use for bilingual models. '
                         'Ignored if embeddings file is given.')
parser.add_argument('--bilingual_en_embed_file', default='',
                    help='Path to a file giving English word vectors in GloVe format. '
                         'If an empty string, randomly initialize and train.')
parser.add_argument('--bilingual_zh_embed_file', default='',
                    help='Path to a file giving Chinese word vectors in GloVe format. '
                         'If an empty string, randomly initialize and train.')


class BilingualGaussianListenerLearner(listener.GaussianContextListenerLearner):
    def get_extra_vars(self):
        id_tag = (self.id + '/') if self.id else ''

        return [T.ivector(id_tag + 'languages')]

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        xs, ys = super(BilingualGaussianListenerLearner, self)._data_to_arrays(
            training_instances=training_instances,
            init_vectorizer=init_vectorizer,
            test=test, inverted=inverted
        )
        langs = [inst.input[0] for inst in training_instances]
        if init_vectorizer:
            self.lang_vec = SymbolVectorizer(use_unk=False)
            self.lang_vec.add_all(langs)
        langs_vec = self.lang_vec.vectorize_all(langs)
        return xs[:1] + [langs_vec] + xs[1:], ys

    def get_embedding_layer(self, l_in, extra_vars):
        language = extra_vars[0]
        context_vars = extra_vars[1:]

        id_tag = (self.id + '/') if self.id else ''

        l_lang = InputLayer(shape=(None,), input_var=language,
                            name=id_tag + 'lang_input')

        if self.options.bilingual_en_embed_file:
            en_embeddings = load_embeddings(self.options.bilingual_en_embed_file, self.seq_vec)
            en_embed_size = en_embeddings.shape[1]
        else:
            en_embeddings = Normal()
            en_embed_size = self.options.bilingual_embed_size

        if self.options.bilingual_zh_embed_file:
            zh_embeddings = load_embeddings(self.options.bilingual_zh_embed_file, self.seq_vec)
            zh_embed_size = zh_embeddings.shape[1]
        else:
            zh_embeddings = Normal()
            zh_embed_size = self.options.bilingual_embed_size

        l_en = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                              output_size=en_embed_size,
                              W=en_embeddings,
                              name=id_tag + 'desc_embed_en')
        print('l_en: {}'.format(l_en.output_shape))
        l_en_transformed = dimshuffle(l_en, (0, 2, 1))
        l_en_transformed = NINLayer(l_en_transformed, num_units=self.options.listener_cell_size,
                                    nonlinearity=None,
                                    name=id_tag + 'desc_embed_en_transformed')
        l_en_transformed = dimshuffle(l_en_transformed, (0, 2, 1))
        print('l_en_transformed: {}'.format(l_en_transformed.output_shape))

        l_zh = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                              output_size=zh_embed_size,
                              W=zh_embeddings,
                              name=id_tag + 'desc_embed_zh')
        print('l_zh: {}'.format(l_zh.output_shape))
        l_zh_transformed = dimshuffle(l_zh, (0, 2, 1))
        l_zh_transformed = NINLayer(l_zh_transformed, num_units=self.options.listener_cell_size,
                                    nonlinearity=None,
                                    name=id_tag + 'desc_embed_zh_transformed')
        l_zh_transformed = dimshuffle(l_zh_transformed, (0, 2, 1))
        print('l_zh_transformed: {}'.format(l_zh_transformed.output_shape))
        l_merged = SwitchLayer(l_lang, [l_en_transformed, l_zh_transformed],
                               name=id_tag + 'desc_embed_switch')
        print('l_merged: {}'.format(l_merged.output_shape))
        return (l_merged, context_vars)

    def unpickle(self, state, model_class=SimpleLasagneModel):
        if isinstance(state, tuple) and isinstance(state[-1], SymbolVectorizer):
            self.lang_vec = state[-1]
            super(BilingualGaussianListenerLearner,
                  self).unpickle(state[:-1], model_class=model_class)
        else:
            self.lang_vec = SymbolVectorizer(use_unk=False)
            self.lang_vec.add_all(['en', 'zh'])
            super(BilingualGaussianListenerLearner,
                  self).unpickle(state, model_class=model_class)


class SwitchLayer(MergeLayer):
    def __init__(self, switch, alternatives, **kwargs):
        super(SwitchLayer, self).__init__([switch] + alternatives, **kwargs)
        assert len(alternatives) >= 1, \
            'Need at least 1 alternative (got {})'.format(len(alternatives))
        self.switch = switch
        self.alternatives = alternatives

    def get_output_shape_for(self, input_shapes):
        assert len(input_shapes) == len(self.alternatives) + 1, 'Need at least '
        assert all(input_shapes[i] == input_shapes[1]
                   for i in range(2, len(input_shapes))), \
            'All alternatives for SwitchLayer must have the same shape; instead ' \
            'got {}.'.format(input_shapes[1:])

        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        switch = inputs[0]
        alternatives = inputs[1:]

        stacked = T.stack(alternatives, axis=0)
        return stacked[switch, T.arange(T.shape(switch)[0]), ...]


def load_embeddings(filename, seq_vec):
    print('Loading word vectors from {}'.format(filename))

    words = []
    vecs = []
    vec_map = {}
    with open(filename, 'r') as infile:
        for line in infile:
            if line:
                line = line.decode('utf-8').split(' ')
                word = line[0]
                vec = np.array([float(e) for e in line[1:]])
                vecs.append(vec)
                words.append(word)
                vec_map[word] = vec

    if '<unk>' not in vec_map:
        if 'UNK' in vec_map:
            unk = vec_map['UNK']
        else:
            num_rare_words = int(len(vecs) / 10) + 1
            unk = np.mean(np.array(vecs[-num_rare_words:]), axis=0)
        vec_map['<unk>'] = unk

    if '<s>' not in vec_map:
        punct_vecs = np.array([vec_map[p] for p in u'([{<"' if p in vec_map])
        start = np.mean(punct_vecs, axis=0)
        vec_map['<s>'] = start

    if '</s>' not in vec_map:
        punct_vecs = np.array([vec_map[p] for p in u'.!?。")]}>' if p in vec_map])
        end = np.mean(punct_vecs, axis=0)
        vec_map['</s>'] = end

    mat = []
    for token in seq_vec.tokens:
        if token in vec_map:
            mat.append(vec_map[token])
        else:
            mat.append(unk)

    return np.array(mat, dtype=theano.config.floatX)


AGENTS = {
    'BilingualListener': BilingualGaussianListenerLearner,
}
