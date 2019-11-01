"""
Ze Xuan Ong
David Bamman
Noah A. Smith

Code for maximum likelihood estimation of a bigram HMM from
column-formatted training data.

Usage:  train_hmm.py tags-file text-file hmm-file

The training data should consist of one line per sequence, with
states or symbols separated by whitespace and no trailing whitespace.
The initial and final states should not be mentioned; they are
implied.

"""

import sys
import re

from collections import defaultdict

# Files
TAG_FILE = sys.argv[1]
TOKEN_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]

# Vocabulary
vocab = {}
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

# Transition and emission probabilities
emissions = {}
bitransitions = {}
bitransitions_total = defaultdict(lambda: 0)
emissions_total = defaultdict(lambda: 0)
tritransitions = {}
tritransitions_total = {}
total_tags = set()
total_tags.add(FINAL_STATE)
total_tags.add(INIT_STATE)

with open(TAG_FILE) as tag_file, open(TOKEN_FILE) as token_file:
    for tag_string, token_string in zip(tag_file, token_file):
        tags = re.split("\s+", tag_string.rstrip())
        tokens = re.split("\s+", token_string.rstrip())
        pairs = zip(tags, tokens)

        prevtag = INIT_STATE
        prevprevtag = INIT_STATE

        for (tag, token) in pairs:

            # this block is a little trick to help with out-of-vocabulary (OOV)
            # words.  the first time we see *any* word token, we pretend it
            # is an OOV.  this lets our model decide the rate at which new
            # words of each POS-type should be expected (e.g., high for nouns,
            # low for determiners).
            total_tags.add(tag)
            if token not in vocab:
                vocab[token] = 1
                token = OOV_WORD

            if tag not in emissions:
                emissions[tag] = defaultdict(lambda: 0)
            if prevtag not in bitransitions:
                bitransitions[prevtag] = defaultdict(lambda: 0)
            if prevprevtag not in tritransitions:
                tritransitions[prevprevtag] = defaultdict(lambda: 0)
            if prevtag not in tritransitions[prevprevtag]:
                tritransitions[prevprevtag][prevtag] = defaultdict(lambda: 0)
            if prevprevtag not in tritransitions_total:
                tritransitions_total[prevprevtag] = defaultdict(lambda: 0)

            # increment the emission/transition observation
            emissions[tag][token] += 1
            emissions_total[tag] += 1
            
            bitransitions[prevtag][tag] += 1
            bitransitions_total[prevtag] += 1

            tritransitions[prevprevtag][prevtag][tag] += 1
            tritransitions_total[prevprevtag][prevtag] += 1

            prevprevtag = prevtag
            prevtag = tag

        # don't forget the stop probability for each sentence
        if prevtag not in bitransitions:
            bitransitions[prevtag] = defaultdict(lambda: 0)
        if prevprevtag not in tritransitions:
            tritransitions[prevprevtag] = defaultdict(lambda: 0)
        if prevtag not in tritransitions[prevprevtag]:
            tritransitions[prevprevtag][prevtag] = defaultdict(lambda: 0)
        if prevprevtag not in tritransitions_total:
            tritransitions_total[prevprevtag] = defaultdict(lambda: 0)

        bitransitions[prevtag][FINAL_STATE] += 1
        bitransitions_total[prevtag] += 1
        tritransitions[prevprevtag][prevtag][FINAL_STATE] += 1
        tritransitions_total[prevprevtag][prevtag] += 1

# deleted interpolation calculation
lambda1 = lambda2 = lambda3 = 0
for a in tritransitions:
    for b in tritransitions[a]:
        for c in tritransitions[a][b]:
            v = tritransitions[a][b][c]
            if v > 0:
                try:
                    c1 = float(v-1)/(bitransitions[a][b]-1)
                except ZeroDivisionError:
                    c1 = 0
                try:
                    c2 = float(bitransitions[a][b]-1)/(emissions_total[a]-1)
                except ZeroDivisionError:
                    c2 = 0
                try:
                    c3 = float(emissions_total[a]-1)/(sum(emissions_total.values())-1)
                except ZeroDivisionError:
                    c3 = 0

                k = max([c1, c2, c3])
                if k == c1:
                    lambda3 += v
                if k == c2:
                    lambda2 += v
                if k == c3:
                    lambda1 += v

weights = [lambda1, lambda2, lambda3]
norm_a = weights[0]/sum(weights)
norm_b = weights[1]/sum(weights)
norm_c = weights[2]/sum(weights)


# Write output to output_file
with open(OUTPUT_FILE, "w") as f:
    num_bitrans_combos = len(total_tags)
    for prevtag in total_tags:
        for tag in total_tags:
            if prevtag in bitransitions and tag in bitransitions[prevtag]:
                f.write("bitrans {} {} {}\n"
                    .format(prevtag, tag, (bitransitions[prevtag][tag]+0.1) / (bitransitions_total[prevtag]+num_bitrans_combos*0.1)))
            elif prevtag in bitransitions:
                f.write("bitrans {} {} {}\n"
                    .format(prevtag, tag, (0.1) / (bitransitions_total[prevtag]+num_bitrans_combos*0.1)))
            else:
                f.write("bitrans {} {} {}\n"
                    .format(prevtag, tag, (0.1) / (num_bitrans_combos*0.1)))

    num_emissions_combos = len(set([token for tag in emissions for token in emissions[tag]]))
    for tag in emissions:
        for token in emissions[tag]:
            f.write("emit {} {} {}\n"
                .format(tag, token, (emissions[tag][token]+0.1) / (emissions_total[tag]+num_emissions_combos*0.1)))

    num_tritrans_combos = sum([sum(tritransitions_total[prevprevtag].values()) for prevprevtag in tritransitions_total])
    for prevprevtag in tritransitions:
        for prevtag in tritransitions[prevprevtag]:
            for tag in tritransitions[prevprevtag][prevtag]:
                f.write("tritrans {} {} {} {}\n"
                    .format(prevprevtag, prevtag, tag, (tritransitions[prevprevtag][prevtag][tag]+0.1) / (tritransitions_total[prevprevtag][prevtag]+num_tritrans_combos*0.1)))
    f.write("norms {} {} {}\n".format(norm_a, norm_b, norm_c))

    total_emissions = sum(emissions_total.values())
    for tag in emissions_total:
        if tag!="init":
            f.write("unitag {} {}\n".format(tag, emissions_total[tag] / total_emissions))

