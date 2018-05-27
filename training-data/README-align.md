Script to get a word-level alignment between reference and hypothesis texts, suitable to use as training data for the bi-LSTM neural model.

command for processing a single pair of reference-hypothesis texts: 
```
python align.py single reference.txt hypothesis.txt > output_file
```

for processing a series of pairs specified in a file containing one white space-separated pair per line: 
```
python align.py batch batchfile.txt > output_file
```
##Output
The format of output_file will be pairs of reference and hypothesis text, separated by one blank line:
START_OF_FILE
reference_line_1
hypothesis_line_1

reference_line_2
hypothesis_line_2

reference_line_3
...