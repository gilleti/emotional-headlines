import os
import pandas as pd
import re

import warnings
warnings.simplefilter("ignore")

def is_dir(paths):
    dirs = []
    for path in paths:
        path = dir + path
        if os.path.isdir(path) == True:
            dirs.append(path)
    return dirs

def get_files(dirs):
    file_list = []
    for dir in dirs:
        files = os.listdir(dir)
        for file in files:
            if file.endswith("xlsx"):
                file_list.append(dir + "/" + file)
    return file_list


# Below are the cleaning functions


def ungrammatical_quotes(df):
    """ Remove headlines with an uneven number of quote signs as these indicate poor quality headlines.
    Also write the deleted headlines to file for later review.
    Input: df
    Output: inplace modified df
    """

    quote_indications = ["\"", "\'", "\”"] 
    
    with open("removed_headlines.txt", "a") as f:
        for index, row in df.iterrows(): 
            headline = str(row["HEADLINE"])
            quotes = 0
            for i in headline:
                if str(i) in quote_indications:
                    quotes += 1
            if (quotes % 2) != 0: # Find strings with an uneven number of quotes
                try:
                    df.drop(index, inplace=True) # Drop them
                    s = "ungrammatical quote:" + headline + "\n"
                    f.write(s)
                except KeyError:
                    continue
    return df

def mostly_numbers(df):
    """Find and drop headlines that contain all numbers and spaces.
    This function could be improved upon by disallowing a ration of numbers to letters.
    Also write the deleted headlines to file for later review.
    Input: df
    Output: inplace modified df
    """
    with open("removed_headlines.txt", "a") as f:
        for index, row in df.iterrows():
            headline = str(row["HEADLINE"])
            if row["HEADLINE"].strip().replace(" ", "").isdigit() == True:
                try:
                    df.drop(index, inplace=True)
                    s = "mostly numbers: " + headline + "\n"
                    f.write(s)
                except KeyError:
                    continue
    return df

def forbidden_words(df):
    """ Removes all headlines that contain a forbidden word.
    These headlines are mostly sports related.
    Also writes removed headlines to file.
    Input: df
    Output: inplace modified df
    
    """
    # Consider line below an alternative
    # df = df[~df.HEADLINE.str.contains('|'.join(completely_forbidden_words))] # Drop rows with forbidden words
    completely_forbidden_words = ["v-5", "v5", "v 5", "v75", "v 75", "formel 1", "formel1", "ingen rubrik", "v65", "v 65", "v64", "v 64", "trav"]

    with open("removed_headlines.txt", "a") as f:
        for index, row in df.iterrows():
            headline = row["HEADLINE"]
            if any(forbidden_word in row["HEADLINE"].lower() for forbidden_word in completely_forbidden_words):
                try:
                    df.drop(index, inplace=True)
                    s = "forbidden words: " + headline + "\n"
                    f.write(s)
                except KeyError:
                    continue
    return df

def tags(df):
    with open("removed_headlines.txt", "a") as f:
        tag_like_chars = ["[", "]", "<", "/", "\\", ".se", ".com", ".nu", "@"]
        for index, row in df.iterrows(): # Find all tag-like headlines
            headline = str(row["HEADLINE"])
            if any(tag in headline.lower() for tag in tag_like_chars):
                try:
                    df.drop(index, inplace=True)
                    s = "tag: " + headline + "\n"
                    f.write(s)
                except KeyError:
                    continue
    return df

def too_long(df):
    # Drop rows where word count is over 25
    with open("removed_headlines.txt", "a") as f:
        for index, row in df.iterrows(): 
            headline = str(row["HEADLINE"])
            if len(headline.strip().split(' ')) > 15:
                try:
                    df.drop(index, inplace=True)
                    s = "too long: " + headline + "\n"
                    f.write(s)
                except KeyError:
                    continue
    return df

def too_short(df):
    with open("removed_headlines.txt", "a") as f:
        for index, row in df.iterrows(): # Find all rows with less than three tokens that do not start with a capitalized letter
            headline = str(row["HEADLINE"])
            if len(headline.strip().split(' ')):
                if headline.strip().split(' ')[0].islower():
                    try:
                        df.drop(index, inplace=True)
                        s = "too short: " + headline + "\n"
                        f.write(s)
                    except KeyError:
                        continue
    return df

def names(df):
    """Find and drop headlines that contain only names, based on a simple regex. 
    The regex could be improved upon.
    Also write removed headlines to file.
    This function adds two columns to the dataframe: 
        - regex_pattern: which documents the detected name
        - regex_pattern_headline_diff: which identifies the number of characters that differ the regex pattern from the headline string as a whole. This is used in the obituary function.

    Input: df
    Output: inplace modified df
    """
    
    # TODO() Write removed headlines to file
    r = r'([A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+)' # Simple Firstname Lastname regex
    df['regex_pattern'] = df['HEADLINE'].str.extract(r, expand=True)
    df["regex_pattern_headline_diff"] = df.HEADLINE.str.len() - df.regex_pattern.str.len()
    with open("removed_headlines.txt", "a") as f:
        for index, row in df.iterrows():
            headline = str(row["HEADLINE"])
            if row["regex_pattern_headline_diff"] == 0:
                try:
                    df.drop(index, inplace=True)
                    s = "name: " + headline + "\n"
                    f.write(s)
                except KeyError:
                    continue
    return df

def has_digit(s):
    return any(i.isdigit() for i in s)

def obituary(df):
    # Doesn't work
    """Find and drop headlines that look like obituary. 
    We do this by looking at a subset of the data: headlines that looks like they include the name, indicated by regex_pattern_headline_diff.
    If the difference between the length of the name and the length of the headline is 11 (e.g "Bertil Karlsson 1943-2011") and the headline contains numbers, drop it.
    Also write removed headlines to file.
    Input: df
    Output: inplace modified df
    """
    with open("removed_headlines.txt", "a") as f:
        for index, row in df.iterrows():
            headline = str(row["HEADLINE"])
            if has_digit(headline):
                if row['regex_pattern_headline_diff'] == 11:
                    try:
                        df.drop(index, inplace=True)
                        s = "obituary: " + headline + "\n"
                        f.write(s)
                    except KeyError:
                        continue
    # TODO() SHould the number really be 11?
    # TODO() This is a time consuming function. Consider re-write.
    return df





dir = "/home/hilhag/prjs/emotional-headlines/headlines/"
paths = os.listdir(dir)
dirs = is_dir(paths)
files = get_files(dirs)

# Read all files
appended_data = []

for file in files:
    print("Processing file: " + file)
    data = pd.read_excel(file)
    data.dropna(subset=['HEADLINE'], how='all', inplace=True)
    data.drop_duplicates(subset=['HEADLINE'], keep='first', inplace=True) # Deduplicate small file
    print("Deduplication of individual file finished.")
    appended_data.append(data)

print("Concatenate files...")
df = pd.concat(appended_data)
print("Files concatenated.")
print("Size of original dataframe is: " + str(len(df)))
df.drop_duplicates(subset=['HEADLINE'], keep='first', inplace=True) # Deduplicate big file
print("Concatenated file deduplicated.")
original_deduplicated_frame_size = len(df)
print("Size of deduplicated dataframe is: " + str(original_deduplicated_frame_size))
print("Writing initial dataframe to file...")
df.to_parquet("headlines_not_filtered.parquet")
print("Wrote to file.")

print("Init ungrammatical quotes.")
df = ungrammatical_quotes(df)
print(len(df))

print("Mostly numbers.")
df = mostly_numbers(df)
print(len(df))

print("Too short.")
df = too_short(df)
print(len(df))

print("Too long.")
df = too_long(df)
print(len(df))

print("Tags.")
df = tags(df)
print(len(df))

print("Forbidden words.")
df = forbidden_words(df)
print(len(df))

print("Names.")
df = names(df)
print(len(df))

print("Obituary.")
df = obituary(df)
print(len(df))

df.to_parquet("headlines.parquet")

print("Size of final frame: " + str(len(df)))
