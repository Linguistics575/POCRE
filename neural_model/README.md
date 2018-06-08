# Pretrained Model
The information that is saved for this model is in the pretrained/ folder. It contains the checkpoint, vocabulary data, and metadata for the pretrained model.

The model takes in your data file, and outputs the corrected string directly.

The command line to run the model is:

```
python run_pretrained-BE-DD.py your_input_file > your_output_file
```

Make sure the the pretrained/bidirectENC_dynamDEC/ folder is in your working directory. No gold standard alignment is needed and no evaluation metrics are printed by this script. The only restriction on the input file is that the lines must not be longer than 133 characters for the pretrained model.

# How To Train Your Model
This encoder decoder neural network model for post OCR correction is based on tensorflow.

To train your own model new model, first create a text file with the path to each training file path on a new line, like the following:

```
\path\to\file1.txt
\path\to\file2.txt
```

Make sure each of these files have been aligned with align.py. Do the same for a single dev test file.

Go into the bidirect_enc_dynam_dec.py and change the output CHECKPOINT variable (around line 341) to your desired save location. You may also change EPOCHS, NUM_IN_BATCH, state_size, or embedding_size if you wish. Note that num_layers is not currently used so changing that will not result in any change in the model at the moment.

Run the following command to train your model. NOTE IT MAY TAKE A DAY OR SO TO TRAIN DEPENDING ON HOW MANY TRAINING FILES YOU INPUT AND THE NUMBER OF EPOCHS YOU TRAIN FOR.

```
python bidirect_enc_dynam_dec.py train_paths_file dev_test_file
```

This will train the model and save it to your desired save folder.

Go into run_pretrained_BE-DD.py and change CHECKPOINT and vocab_idx_file to the appropriate locations that you declared in bidirect_enc_dynam_dec.py. You can now use driver.sh to run future test data on your trained model!

Future work will include a configuration file to modify training and checkpoint parameters to reduce code changing and duplication.
