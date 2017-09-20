from color_instances import filtered_dev, chinese_dev
from tokenizers import heuristic_ending_tokenizer as en_tokenizer
from tokenizers import chinese_tokenizer as zh_tokenizer


def calculate_oov_rate(data, tokenizer, wordvecs):
    tokens = [w for inst in data for w in tokenizer(inst.input)]
    types = set(tokens)
    vocab = set(get_vocab(wordvecs))
    oov_type = len(types - vocab) * 1.0 / len(types)
    oov_token = len([w for w in tokens if w not in vocab]) * 1.0 / len(tokens)
    return oov_type, oov_token


def get_vocab(wordvecs):
    vocab = []
    with open(wordvecs, 'r') as infile:
        for line in infile:
            if line.strip():
                vocab.append(line[:line.index(' ')])
    return vocab


if __name__ == '__main__':
    en_data = filtered_dev(listener=True)
    print('GloVe: {0} type, {1} token'.format(
        *calculate_oov_rate(en_data, en_tokenizer, 'word_vectors/glove_en_50d.txt')
    ))
    zh_data = chinese_dev(listener=True)
    print('Chinese: {0} type, {1} token'.format(
        *calculate_oov_rate(zh_data, zh_tokenizer, 'word_vectors/will_zh_50d.txt')
    ))
