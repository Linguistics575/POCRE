# POCRE
POCRE (“poker”) - Post Optical Character Recognition Evaluation

## Description
POCRE is a system for correcting text that is the output of an OCR system. OCR systems are imperfect; depending on how they're trained and the quality of the images being processed, they can make errors interpreting characters from pixels. These errors differ from human errors because they are based on visual characteristics rather than meaning or sound. The purpose of this system is to indicate and correct likely machine-introduced errors in OCR-produced which for a human annotator can then review.

## Modeling
The system is based on a bidirectional LSTM neural network built in Tensorflow (https://www.tensorflow.org/). We provide users with a pretrained model that can be imported for testing new data. Please note, however, that the system will perform best when it has been trained on data that is similar to the data it will be used on, in such characteristics as the approximate number of errors, the language of the text, and the general content/lexicon. The data used to train our pretrained model, consisting of typewritten historical English text focusing on Egyptian archaeology, is provided so that users can decide whether the pretrained model is suitable for their needs.

## Installation and Running the Model on Your Machine
For detailed instructions on installing Tensorflow, please refer to https://www.tensorflow.org/install/. 

The README under neural_model/ provides specific command line instructions.