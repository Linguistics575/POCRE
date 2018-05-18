# POCRE
POCRE (“poker”) - Post Optical Character Recognition Evaluation
Optical Character Recognition (OCR) - retreiving text from images automatically

## Description
POCRE is a system for correcting text that is the output of an OCR system. OCR systems, depending on how their trained and the quality of the images that are input, frequently make errors interpreting characters from pixels. These errors differ from human errors because they are often based on visual characteristics rather than meaning or sound. The purpose of this system is to predict machine-introduced errors for a human annotator to later correct for any relevant downstream task.

## Modeling
The model is based on a bidirectional LSTM neural net built in Tensorflow (https://www.tensorflow.org/). We provide users with a pretrained model that can be imported for testing novel data. Please note, however, that the model will perform best when it has been trained on data that is similar to the test data. This includes details such as the estimated number of errors, the language of the text, and the general content. Our training data, consisting of typewritten historical English text focusing on Egyptian archaeology, is provided so that users can decide whether the pretrained model is sufficient for their needs. Please also note that this system will not recover files that are beyond human recognition (e.g. $#24*^.....*).

## Installation and Running the Model on Your Machine
For detailed instructions on installing Tensorflow, please refer to https://www.tensorflow.org/install/. 

The README under neural_model/ provide specific command line instructions.

