"""
Yihui Peng
Ze Xuan Ong
Jocelyn Huang
Noah A. Smith

Usage: python viterbi.py <HMM_FILE> <TEXT_FILE> <OUTPUT_FILE>

Apart from writing the output to a file, the program also prints
the number of text lines read and processed, and the time taken
for the entire program to run in seconds. This may be useful to
let you know how much time you have to get a coffee in subsequent
iterations.

"""

import math
import sys
import time
import itertools

from collections import defaultdict

# Magic strings and numbers
HMM_FILE = sys.argv[1]
TEXT_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]
BIGRAM_TRANSITION_TAG = "bitrans"
TRIGRAM_TRANSITION_TAG = "tritrans"
NORM_TAG = "norms"
EMISSION_TAG = "emit"
UNIGRAM_TAG = "unitag"
OOV_WORD = "OOV"         # check that the HMM file uses this same string
INIT_STATE = "init"      # check that the HMM file uses this same string
FINAL_STATE = "final"    # check that the HMM file uses this same string

# Transition and emission probabilities
# Structured as a nested defaultdict in defaultdict, with inner defaultdict
#   returning 0.0 as a default value, since dirty KeyErrors are equivalent to
#   zero probabilities
#
# The advantage of this is that one can add redundant transition probabilities
bitransition = defaultdict(lambda: defaultdict(lambda: 1.0))
tritransition = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 1.0)))
emission = defaultdict(lambda: defaultdict(lambda: 1.0))
uniform = defaultdict(lambda: 0.0)

# Store states to iterate over for HMM
# Store vocab to check for OOV words
states = set()
state_pairs = set()
vocab = set()
norm_a = norm_b = norm_c = None

# original bigram viterbi algorithm
def bigram_viterbi(index, line):
    words = line.split()

    # Setup Viterbi for this sentence:
    # V[x][y] where x is the index of the word in the line
    #   and y is a state in the set of states
    V = defaultdict(lambda: {})

    # Initialize backtrace so we get recover the path:
    # back[x][y] where x is the index of the word in the line
    #   and y is the previous state with the highest probability
    back = defaultdict(lambda: defaultdict(lambda: ""))

    # Initialize V with init state
    # REDUNDANT
    V[-1][INIT_STATE] = 0.0

    # Iterate over each word in the line
    for i in range(len(words)):

        # If word not in vocab, replace with OOV word
        if words[i] not in vocab:
            words[i] = OOV_WORD

        # Iterate over all possible current states:
        for state in states:

            # If this emission is impossible for this state, move on
            emission_prob = emission[state][words[i]]
            if emission_prob > 0.0 or emission_prob == 1.0:
                continue

            # Iterate over all possible previous states:
            # Specifically, prev_state -> state
            for prev_state in states:

                # If this transition is impossible for this state, move on
                bitransition_prob = bitransition[prev_state][state]
                if bitransition_prob > 0.0 or bitransition_prob == 1.0:
                    continue

                # Calculate the (log) probability of:
                # 1. The highest probability of the previous state being pre_state
                # 2. The probability of transition prev_state -> state
                # 3. The probability of emission state -> words[i]
                try:
                    v = V[i - 1][prev_state] + bitransition_prob + emission_prob
                except KeyError as e:
                    continue

                # Replace if probability is higher
                current = V[i].get(state, None)
                if not current or v > V[i][state]:
                    V[i][state] = v
                    back[i][state] = prev_state

    # Handle final state    
    best_final_state = None
    for state in states:
        bitransition_prob = bitransition[state][FINAL_STATE]
        if bitransition_prob >= 0:
            continue

        try:
            v = V[len(words) - 1][state] + bitransition_prob
        except KeyError as e:
            continue

        # Replace if probability is higher
        current = V[len(words)].get(FINAL_STATE, None)
        if not current or v > V[len(words)][FINAL_STATE]:
            V[len(words)][FINAL_STATE] = v
            best_final_state = state

    return (best_final_state, back)

# includes trigram and deleted interpolation
def trigram_viterbi(index, line):
    words = line.split()

    # Setup Viterbi for this sentence:
    # V[x][y] where x is the index of the word in the line
    #   and y is a state pair in the set of state pairs
    V = defaultdict(lambda: {})

    # Initialize backtrace so we get recover the path:
    # back[x][y] where x is the index of the word in the line
    #   and y is the previous state pair with the highest probability
    back = defaultdict(lambda: defaultdict(lambda: ""))

    # Initialize V with init state pair
    # REDUNDANT
    V[-1] = defaultdict(lambda: 0.0)
    V[-1][(INIT_STATE, INIT_STATE)] = 1.0

    # Iterate over each word in the line
    for i in range(len(words)):

        # If word not in vocab, replace with OOV word
        if words[i] not in vocab:
            words[i] = OOV_WORD

        # Iterate over all possible current states:
        for state in states:

            # If this emission is impossible for this state, move on
            emission_prob = emission[state][words[i]]
            if emission_prob > 0.0 or emission_prob == 1.0:
                continue

            # Iterate over all possible previous states:
            # Specifically, prev_state -> state
            for (prev_prev_state, prev_state) in state_pairs:

                # If this emission is impossible for the prev state, move on
                if i>0:
                    prev_emission_prob = emission[prev_state][words[i-1]]
                    if prev_emission_prob > 0.0 or prev_emission_prob == 1.0:
                        continue
                # If this emission is impossible for the prev prev state, move on
                if i>1:
                    emission_prob = emission[state][words[i-2]]
                    if emission_prob > 0.0 or emission_prob == 1.0:
                        continue
                # If this transition is impossible for this state, move on
                tritransition_prob = math.exp(tritransition[prev_prev_state][prev_state][state])
                bitransition_prob = math.exp(bitransition[prev_state][state])
                uniform_prob = math.exp(uniform[state])
                # interpolated_prob = math.log(norm_a*uniform_prob + norm_b*bitransition_prob + norm_c*tritransition_prob)
                interpolated_prob = math.log(tritransition_prob)
                if interpolated_prob > 0.0 or interpolated_prob == 1.0:
                    continue

                # Calculate the (log) probability of:
                # 1. The highest probability of the previous state being pre_state
                # 2. The probability of transition prev_state -> state
                # 3. The probability of emission state -> words[i]
                try:
                    v = V[i - 1][(prev_prev_state, prev_state)] + interpolated_prob + emission_prob
                except KeyError as e:
                    continue

                # Replace if probability is higher
                current = V[i].get((prev_state, state), None)
                if not current or v > V[i][(prev_state, state)]:
                    V[i][(prev_state, state)] = v
                    back[i][state] = prev_state

    # Handle final state    
    best_final_state = None
    for (prev_state, state) in state_pairs:
        tritransition_prob = math.exp(tritransition[prev_state][state][FINAL_STATE])
        bitransition_prob = math.exp(bitransition[state][FINAL_STATE])
        uniform_prob = 0
        # interpolated_prob = math.log(norm_a*uniform_prob + norm_b*bitransition_prob + norm_c*tritransition_prob)
        interpolated_prob = math.log(tritransition_prob)
        if interpolated_prob >= 0:
            continue

        try:
            v = V[len(words) - 1][(prev_state, state)] + interpolated_prob
        except KeyError as e:
            continue

        # Replace if probability is higher
        current = V[len(words)].get((state, FINAL_STATE), None)
        if not current or v > V[len(words)][(state, FINAL_STATE)]:
            V[len(words)][(state, FINAL_STATE)] = v
            best_final_state = state
    return (best_final_state, back)


# Actual Viterbi function that takes a list of lines of text as input
# The original version (and most versions) take in a single line of text.
# This is to reduce process creation/tear-down overhead by allowing us
#   to chunk up the input and divide it amongst the processes without
#   resource sharing
# NOTE: the state and vocab sets are still shared but it does not seem
#       to impact performance by much
def viterbi(lines):
    ret = [""] * len(lines)
    for (index, line) in enumerate(lines):
        (bi_best_final_state, bi_back) = bigram_viterbi(index, line)
        bi_sequence = None
        # Backtrace from the best_final_state
        if bi_best_final_state:
            words = line.split()
            output = []
            # Step from len(words) to 0
            for i in range(len(words) - 1, -1, -1):
                output.append(bi_best_final_state)
                bi_best_final_state = bi_back[i][bi_best_final_state]

            # Reverse the output and join as string
            bi_sequence = " ".join(output[::-1])
        # If no best_final_state e.g. could not find transition to terminate
        # then return empty string
        else:
            bi_sequence = ""

        # (tri_best_final_state, tri_back) = trigram_viterbi(index, line)
        # tri_sequence = None
        # # Backtrace from the best_final_state
        # if tri_best_final_state:
        #     words = line.split()
        #     output = []
        #     # Step from len(words) to 0
        #     for i in range(len(words) - 1, -1, -1):
        #         output.append(tri_best_final_state)
        #         tri_best_final_state = tri_back[i][tri_best_final_state]

        #     # Reverse the output and join as string
        #     tri_sequence = " ".join(output[::-1])
        # If no best_final_state e.g. could not find transition to terminate
        # then return empty string
        # else:
        #     tri_sequence = ""
        # if tri_best_final_state:
        #     ret[index] = tri_sequence
        # else:
        #     ret[index] = bi_sequence
        ret[index] = bi_sequence
    # Return a list of processed lines
    return ret




# Main method
def main():

    # Mark start time
    t0 = time.time()

    # Read HMM transition and emission probabilities
    with open(HMM_FILE, "r") as f:
        for line in f:
            line = line.split()

            # Read transition
            # Example line: trans NN NNPS 9.026968067100463e-05
            # Read in states as prev_state -> state
            if line[0] == BIGRAM_TRANSITION_TAG:
                (prev_state, state, trans_prob) = line[1:4]
                bitransition[prev_state][state] = math.log(float(trans_prob))
                states.add(prev_state)
                states.add(state)

            # Read in states as state -> word
            elif line[0] == EMISSION_TAG:
                (state, word, emit_prob) = line[1:4]
                emission[state][word] = math.log(float(emit_prob))
                states.add(state)
                vocab.add(word)

            elif line[0] == TRIGRAM_TRANSITION_TAG:
                (prev_prev_state, prev_state, state, trans_prob) = line[1:5]
                tritransition[prev_prev_state][prev_state][state] = math.log(float(trans_prob))
                states.add(prev_prev_state)
                states.add(prev_state)
                states.add(state)
            elif line[0] == NORM_TAG:
                (norm_a, norm_b, norm_c) = line[1:4]
            elif line[0] == UNIGRAM_TAG:
                (state, state_prob) = line[1:3]
                uniform[state] = math.log(float(state_prob))


    # Create the set of all state pairs
    non_initial_states = states - set(INIT_STATE)
    non_initial_state_pairs = set([p for p in itertools.product(non_initial_states, repeat=2)])
    union_with = set([(INIT_STATE, state) for state in non_initial_states]) | non_initial_state_pairs
    state_pairs.update(union_with)
    state_pairs.add((INIT_STATE, INIT_STATE))

    # Read lines from text file and then split by number of processes
    text_file_lines = []
    with open(TEXT_FILE, "r") as f:
        text_file_lines = f.readlines()

    results = viterbi(text_file_lines)

    # Print output to file
    with open(OUTPUT_FILE, "w") as f:
        for lines in results:
            for line in lines:
                f.write(line)
            f.write("\n")

    # Mark end time
    t1 = time.time()

    # Print info to stdout
    print("Processed {} lines".format(len(text_file_lines)))
    print("Time taken to run: {}".format(t1 - t0))

if __name__ == "__main__":    
    main()
