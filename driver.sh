#!/bin/bash
# run from main "POCRE" directory
# sample command: sh driver.sh input_file.txt output_file --show_original 

# $1 is raw OCR text (system input)
# $... are flags (--numbered or --show_original)

python neural_model/run_pretrained_BE-DD.py $1 | python show_changes.py $@
