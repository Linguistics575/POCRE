# Pretrained Model
The information that is saved for this model is in the pretrained/ folder. It contains the checkpoint, vocabulary data, and metadata for the pretrained model.

The model takes in your data file, and outputs the corrected string directly.

The command line to run the model is:
##python run_pretrained.py your_data_file

Make sure the the pretrained/ folder is in your working directory. No gold standard is needed and no evaluation metrics are printed by this script.

# How To Train Your Model
This neural network model for post OCR correction is based on tensorflow.

Takes in 2 files as arguments. The first file is the training data, and the second file is the test data. 
The program reads in the files, builds a neural model based on Tensorflow, dynamically trains on the training data
in batches for a number of epochs (as determined by the EPOCHS variable), and test the trained model on the test data.

Data is expected to be in the following format and aligned by character:
<Start of file>
Gold standard line 1
OCR output line 1

Gold standard line 2
OCR output line 2

...
Gold standard line x
OCR output line x
<End of file>

Note that the program will crash if the data is not aligned by character!

Example command line:
##python train_lstm.py training_file test_file

Note that this program also outputs model building and model training times and an evaluation metric, in addition to the corrected text.
