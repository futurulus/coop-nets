from utils import *
import argparse

def compare(attribute):
    '''
    Performs a full comparison across Chinese and English for a given
    attribute. This includes generating the csv files and plotting the data.
    '''
    generate_csv(attribute, 'zh')
    generate_csv(attribute, 'en')
    f_zh = 'data/{}_zh.csv'.format(attribute)
    f_en = 'data/{}_en.csv'.format(attribute)
    plot_csvs(f_zh, f_en, PLOTTYPE(attribute),
                   'plots/%s_SEABORN.png' % attribute,
                   ylabel=YLABEL(attribute),
                   title=PLOTTITLE(attribute))

if __name__ == '__main__':
    ATTRIBUTES = ['tokens', 'dialogue', 'superlative',
                  'comparative', 'negation', 'specificity']
    parser = argparse.ArgumentParser(description='compare Chinese and English')
    parser.add_argument('-m', '--mode', help='compare, generate, or plot',
                        default=None, choices=['C', 'G', 'P'])
    parser.add_argument('-a', '--attribute', help='choose attributes', # nargs='+',
                        default=None, choices=ATTRIBUTES)
    parser.add_argument('-l','--language', help='choose language',
                        default=None, choices=['zh', 'en'])
    args = parser.parse_args()

    if args.mode == 'C':
        print 'Comparing {}...'.format(args.attribute)
        compare(args.attribute)
        print 'Done.'
    elif args.mode == 'G':
        print 'Generating csv...'
        generate_csv(args.attribute, args.language)
        print 'Done.'
    elif args.mode == 'P':
        f_zh = 'data/{}_zh.csv'.format(args.attribute)
        f_en = 'data/{}_en.csv'.format(args.attribute)
        print 'Plotting...'
        plot_csvs(f_zh, f_en, PLOTTYPE(args.attribute),
                       'plots/%s_SEABORN.png' % args.attribute,
                       ylabel=YLABEL(args.attribute),
                       title=PLOTTITLE(args.attribute))
        print 'Done.'
