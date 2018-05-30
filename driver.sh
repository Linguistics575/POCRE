#!/bin/bash
# run from main "POCRE" directory
# sample command: sh driver.sh input_file.txt --show_original > output_file 

# $1 is raw OCR text (system input)
# $... are flags (--numbered or --show_original)

python neural_model/run_pretrained_BE-DD.py $1 | python evaluation/show_changes.py $@
