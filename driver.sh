#!/bin/bash
# sample command: sh driver.sh input_file.txt --show_original > output_file 
# This enforces single file processing (with the "single" flag in the command) for now, until I fix the batch method in show_changes.py

# $1 is raw OCR text (system input)
# $... are flags

python run_pretrained.py $1 | python show_changes.py single $@
