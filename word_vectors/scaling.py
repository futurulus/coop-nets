import numpy as np
import matplotlib.pyplot as plt


def plot_scaling(filename):
    vecs = load_vectors(filename)
    means = np.mean(vecs, axis=0)
    stds = np.std(vecs, axis=0)
    plt.scatter(means, stds)
    plt.show()


def load_vectors(filename):
    rows = []
    with open(filename, 'r') as infile:
        for line in infile:
            if line.strip():
                try:
                    row = [float(e) for e in line.split(' ')[1:]]
                    rows.append(row)
                    assert len(row) == len(rows[0]), line
                except:
                    print line
                    raise
    return np.array(rows)


if __name__ == '__main__':
    import sys
    plot_scaling(sys.argv[1])
