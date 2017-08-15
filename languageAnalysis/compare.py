from utils import *
import argparse

def plot(a):
    f_zh = 'data/{}_zh.csv'.format(a)
    f_en = 'data/{}_en.csv'.format(a)
    print 'Plotting for {}...'.format(a)
    plot_csvs(f_zh, f_en, PLOTTYPE(a), 'plots/%s.png' % a,
              ylabel=YLABEL(a), title=PLOTTITLE(a))

def compare(attribute):
    '''
    Performs a full comparison across Chinese and English for a given
    attribute. This includes generating the csv files and plotting the data.
    '''
    generate_csv(attribute, 'zh')
    generate_csv(attribute, 'en')
    plot(attribute)

if __name__ == '__main__':
    ATTRIBUTES = ['tokens', 'dialogue', 'superlative',
                  'comparative', 'negation', 'specificity']
    parser = argparse.ArgumentParser(description='compare Chinese and English')
    parser.add_argument('--mode', help='compare, generate, or plot',
                        default=None, choices=['compare', 'csv', 'plot'])
    parser.add_argument('--attributes', help='choose attributes',
                        nargs='*', type=str, default=None,
                        choices=ATTRIBUTES+['all'])
    parser.add_argument('--language', help='choose language',
                        default=None, choices=['zh', 'en'])
    args = parser.parse_args()

    attrs = ATTRIBUTES if args.attributes == ['all'] else args.attributes
    if args.mode == 'compare':
        for a in attrs:
            print 'Comparing {}...'.format(a)
            compare(a)
        print 'Done.'
    elif args.mode == 'csv':
        for a in attrs:
            print 'Generating csv for {}...'.format(a)
            generate_csv(a, args.language)
        print 'Done.'
    elif args.mode == 'plot':
        for a in attrs:
            plot(a)
        print 'Done.'
