#!/usr/bin/env python
"""
Clean up the data
1 - Removes messages greater than 4 standard deviations away from the mean (do this first)
2 - Remove the spam messages :c
Author: Andrew
"""
import pandas as pd


INPUT_FILE = 'colorReferenceMessageChinese.csv'
MESSAGE_COLUMN = 'contents'
SEPARATOR = ','
NUM_COLUMNS = 5

def main():
    """
    Performs preprocessing (cleanup) on the chinese 
    """
    # remove extra commas from the messages
    stripped_file_name = remove_extra_separators(INPUT_FILE)
    opened_stripped_f = open(stripped_file_name, 'r')

    # read into a pandas df
    raw_df = pd.read_csv(opened_stripped_f, sep=SEPARATOR)
    raw_num_rows = raw_df.shape[0]
    print 'Original num datapoints: ', raw_num_rows

    # filter out the long messages and get information
    filtered_length_df, removed_df, mean, std = filter_long_messages(raw_df)
    filtered_num_rows = filtered_length_df.shape[0]

    # print summary results
    print 'Filtered by length num datapoints: ', filtered_num_rows
    print 'Difference: ', raw_num_rows - filtered_num_rows
    print 'Mean: {}, std: {}'.format(mean, std)

    # write to file
    filtered_out_name = INPUT_FILE[:-4] + '_filtered.csv'
    print 'Writing filtered to ', filtered_out_name
    filtered_length_df.to_csv(filtered_out_name, encoding='utf-8')
    
    removed_out_name = 'filtered_removed_messages.csv'
    print 'Writing removed messages to ', removed_out_name
    removed_df.to_csv(removed_out_name, encoding='utf-8')


def remove_extra_separators(input_file):
    """
    The chinese messages corpus is only supposed to have 5 columns per datapoint. 
    Any extra separators that occur in the dialog needs to be removed for the file 
    to be read correctly. The number of columns is set with the constant NUM_COLUMNS.

    :param input_file: the raw converted csv file from jsonToCSV.py
    :returns: name of the file stripped of extra commas
    """
    f = open(input_file, 'r')
    out_f = open('stripped_extra_commas.csv', 'w')
    for line in f:
        split = line.split(SEPARATOR)
        if len(split) > NUM_COLUMNS:
            # gameid, epochTime, roundNum, role, message,could,have,extra,separators
            num_seps = NUM_COLUMNS - 1
            new_line = SEPARATOR.join(split[slice(num_seps)]) + ' '.join(split[slice(num_seps, len(split))])
        else:
            new_line = SEPARATOR.join(split)
        out_f.write(new_line)

    f.close()
    return out_f.name

def filter_long_messages(raw_df, std=4):
    """
    Removes messages with lengths GREATER than the number of standard deviations given.

    :param df: pandas dataframe
    :param std: number of standard deviations

    :returns: the filtered df, the rows that were removed, the mean, the std
    """
    # We have to use msg.decode('utf-8') so that chinese characters are properly evaluated as 1 character.
    # Technically this means that messages with english will be more likely to be thrown out, as spaces and letters
    # are single characters. This may or may not be an issue.
    raw_df['raw_lengths'] = raw_df[MESSAGE_COLUMN].astype(str).apply(lambda msg: len(msg.decode('utf-8')))
    length_mean = raw_df['raw_lengths'].mean()
    length_std = raw_df['raw_lengths'].std()
    raw_df['z_scores'] = (raw_df['raw_lengths'] - length_mean) / length_std
    # keep based on z_score (the num of standard deviations)
    filtered_length_df = raw_df[raw_df['z_scores'] <= std]
    removed_rows_df = raw_df[raw_df['z_scores'] > std]

    # remove the columns we added for calculating and filtering
    filtered_length_df.drop['raw_lengths']
    filtered_length_df.drop['z_scores']
    removed_rows_df.drop['raw_lengths']
    removed_rows_df.drop['z_scores']

    return filtered_length_df, removed_rows_df, length_mean, length_std



if __name__ == '__main__':
    main()

