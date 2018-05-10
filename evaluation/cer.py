import sys
# set to whatever character is used for 'null' in the given alignment file
NULL = '='

with open(sys.argv[1], 'r') as alignment_file:
    lines = alignment_file.readlines()

    # skip blank or comment lines in the beginning of the file
    n = 0
    while lines[n] == '\n' or lines[n].startswith('#'):
        n += 1

    # tally errors and correct characters
    total = 0
    incorrect = 0

    # compare each character in each line
    for i in range(n + 1, len(lines), 3):
        ref = lines[i - 1].rstrip()
        hyp = lines[i].rstrip()

        assert len(ref) == len(hyp), (ref + '\n' + hyp)

        for index in range(len(ref)):
            r = ref[index]
            h = hyp[index]

            if r != h:
                incorrect += 1
            total += 1

# calculate error rate
print("character error rate: " + str(incorrect / total))