# POCRE
POCRE (“poker”) - Post Optical Character Recognition Evaluation

## Description
POCRE is a system for correcting text that is the output of an OCR system. OCR systems are imperfect; depending on how they're trained and the quality of the images being processed, they can make errors interpreting characters from pixels. These errors differ from human errors because they are based on visual characteristics rather than meaning or sound. The purpose of this system is to indicate and correct likely machine-introduced errors in OCR-produced which for a human annotator can then review.

## Modeling
The system is based on a sequence to sequence neural network built in Tensorflow (https://www.tensorflow.org/). We provide users with a pretrained model that can be imported for testing new data. Please note, however, that the system will perform best when it has been trained on data that is similar to the data it will be used on, in such characteristics as the approximate number of errors, the language of the text, and the general content/lexicon. The data used to train our pretrained model, consisting of typewritten historical English text focusing on Egyptian archaeology, is provided so that users can decide whether the pretrained model is suitable for their needs. We used three different OCR systems to make our training data: LIOS, ABBYY, and Microsoft's Azure. The system performance may also be affected by the OCR system that produces the input data.

## Installation and Running the Model on Your Machine
This system requires installations of Python 3 and Tensorflow (developed in version 1.5, not guaranteed to be compatible with other versions).
For detailed instructions on installing Tensorflow, please refer to https://www.tensorflow.org/install/. 

The system can be run from the command line with the following command:

```
sh driver.sh input_file.txt output_file.rtf [--show_original] [--numbered] 
```
input_file.txt is the text you wish to have corrected 
output_file.rtf is the destination for the results
The brackets around --show_original and --numbered indicate that they are optional.
--show_original will display the input text in a column to the right of the corrected output text.
--numbered will print line numbers in the left margin of the output file.

## Copyright Notice
This code is licensed under the MIT License. See LICENSE.txt for more information.

## Issues
See https://github.com/Linguistics575/POCRE/wiki/Technical-Documentation for details about current issues and future work.
