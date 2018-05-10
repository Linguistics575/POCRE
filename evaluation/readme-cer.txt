A script to calculate character error rate. Given a file containing a one-to-one character alignment of a reference text and an OCR hypothesis text, it tallies mismatched characters and matching characters and calculates an error rate. Since the alignment is one-to-one (null characters allowing for one-to-many mappings), this tally accounts for insertions and deletions as well as substitutions.

The null character can be changed from '=' to whatever character is used in the alignment file in line 3.

Command (using python 3): "python cer.py alignment_file"