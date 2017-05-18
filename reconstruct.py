import numpy as np
from scipy.misc import logsumexp


def input_and_show_tables():
    print('Copy-and-paste L0 and S0 from s0_grids, as a dict {"L0": [[...]], "S0": [[...]]}. ')
    print('Press ^D when done.')
    s0_dict_lines = []
    while True:
        try:
            line = raw_input('')
        except EOFError:
            break
        s0_dict_lines.append(line)
    s0_dict = eval(''.join(line.strip() for line in s0_dict_lines))

    utts_str = raw_input('Enter rows of the table to select utts from, separated by comma:\n')
    utts = [int(i) for i in utts_str.strip().split(',')]
    cols_str = raw_input('Enter order of cols of the table separated by comma:\n')
    cols = [int(i) for i in cols_str.strip().split(',')]
    show_tables(s0_dict, utts, cols)


def show_tables(s0_dict, utts, cols):
    sw = 0.608
    bw = -0.15
    alpha = 0.544
    gamma = 0.509

    l0 = np.array(s0_dict['L0']).T[utts, :][:, cols]
    print('L0:\n{}'.format(friendly(l0)))
    print('')

    scaled = l0 * alpha
    s1 = norm(scaled, axis=0)
    print('S1:\n{}'.format(friendly(s1)))
    print('')

    l2 = norm(s1)
    print('L2:\n{}'.format(friendly(l2)))
    print('')

    s0 = np.array(s0_dict['S0']).T[0][cols]
    print('S0:\n{}'.format(np.exp(s0)))
    print('')

    l1 = norm(s0)
    print('L1:\n{}'.format(friendly(l1)))
    print('')

    la = norm((1-sw) * l0[0] + sw * l1)
    print('La:\n{}'.format(friendly(la)))
    print('')

    lb = norm(bw * l0[0] + (1-bw) * l2[0])
    print('Lb:\n{}'.format(friendly(lb)))
    print('')

    le = norm(gamma * lb + (1-gamma) * la)
    print('Le:\n{}'.format(friendly(le)))
    print('')


def friendly(arr):
    return np.round(np.exp(arr), 3)


def norm(arr, axis=-1):
    return arr - logsumexp(arr, axis=axis, keepdims=True)


if __name__ == '__main__':
    input_and_show_tables()
