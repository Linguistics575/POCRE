Script to get a printed word-level alignment between reference and hypothesis texts.

command for processing a single pair of reference-hypothesis texts: "python align.py single <reference.txt> <hypothesis.txt>

for processing a series of pairs specified in a file containing one white space-separated pair per line: "python align.py batch <batchfile.txt>"

The default alignment is horizontal; to get a vertical alignment, add the vertical flag to either a single or batch command:
"python align.py single [-vertical | -v] <ref.txt> <hypo.txt>"

