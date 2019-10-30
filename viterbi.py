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

from collections import defaultdict

# Magic strings and numbers
HMM_FILE = sys.argv[1]
TEXT_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]
TRANSITION_TAG = "trans"
EMISSION_TAG = "emit"
OOV_WORD = "OOV"         # check that the HMM file uses this same string
INIT_STATE = "init"      # check that the HMM file uses this same string
FINAL_STATE = "final"    # check that the HMM file uses this same string

# Transition and emission probabilities
# Structured as a nested defaultdict in defaultdict, with inner defaultdict
#   returning 0.0 as a default value, since dirty KeyErrors are equivalent to
#   zero probabilities
#
# The advantage of this is that one can add redundant transition probabilities
transition = defaultdict(lambda: defaultdict(lambda: 1.0))
emission = defaultdict(lambda: defaultdict(lambda: 1.0))

# Store states to iterate over for HMM
# Store vocab to check for OOV words
states = set()
vocab = set()


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
                    transition_prob = transition[prev_state][state]
                    if transition_prob > 0.0 or transition_prob == 1.0:
                        continue

                    # Calculate the (log) probability of:
                    # 1. The highest probability of the previous state being pre_state
                    # 2. The probability of transition prev_state -> state
                    # 3. The probability of emission state -> words[i]
                    try:
                        v = V[i - 1][prev_state] + transition_prob + emission_prob
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
            transition_prob = transition[state][FINAL_STATE]
            if transition_prob >= 0:
                continue

            try:
                v = V[len(words) - 1][state] + transition_prob
            except KeyError as e:
                continue

            # Replace if probability is higher
            current = V[len(words)].get(FINAL_STATE, None)
            if not current or v > V[len(words)][FINAL_STATE]:
                V[len(words)][FINAL_STATE] = v
                best_final_state = state

        # Backtrace from the best_final_state
        if best_final_state:

            output = []
            # Step from len(words) to 0
            for i in range(len(words) - 1, -1, -1):
                output.append(best_final_state)
                best_final_state = back[i][best_final_state]

            # Reverse the output and join as string
            ret[index] = " ".join(output[::-1])

        # If no best_final_state e.g. could not find transition to terminate
        # then return empty string
        else:
            ret[index] = ""

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
            if line[0] == TRANSITION_TAG:
                (prev_state, state, trans_prob) = line[1:4]
                transition[prev_state][state] = math.log(float(trans_prob))
                states.add(prev_state)
                states.add(state)

            # Read in states as state -> word
            elif line[0] == EMISSION_TAG:
                (state, word, emit_prob) = line[1:4]
                emission[state][word] = math.log(float(emit_prob))
                states.add(state)
                vocab.add(word)


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
