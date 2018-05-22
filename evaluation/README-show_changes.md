A program to show where an edited text (such as the output from an OCR correction system) differs from the original (such as the raw OCR'd text). 

<<<<<<< HEAD
# Input
The original and edited text files, in plain text format (.txt). It is assumed that the files have line-by-line correspondence (line breaks are the same).

# Output 
The edited text (in .rtf format) with changes in bold and red. Words that appear in the original text but not in the edited text appear inside square brackets: "\[\]" (see example output below). The rich text file format should be openable by your operating system's basic text editor or in Google Drive. 
=======
## Input
The hypothesis and reference text files, in plain text format (.txt). It is assumed that the files have line-by-line correspondence (line breaks are the same).

## Output 
The hypothesis text with changes shown in bold and red. Words that appear in the reference text but not in the hypothesis text are preceded by "DELETED: "
>>>>>>> 5c1288050b73ce067a911693d785b668f7f762b6

> __[:]__ The boisterous Mor Whymper came to __breakfast. .....__ I've decided all that 
> excitable volubility in his company __B......__ er because when he crosses over 
> with Don's and me in the felucca on our way to church he is __Euite__ pleasantly 


To run from the command line: 
* Make sure you have python installed. (You can check by opening your command line utility (Terminal for Macs, Command Prompt for Windows) and typing "python".) 
* Navigate to the directory that contains the show_changes.py file.
* Run the following command:
<<<<<<< HEAD
'''
python show_changes.py single path/reference_file.txt path/hypothesis_file.txt > output_file.rtf
'''
=======
```
python show_changes.py single path/reference_file.txt path/hypothesis_file.txt > output_file.txt
```
>>>>>>> 5c1288050b73ce067a911693d785b668f7f762b6

## NOTE
This process will eventually be incorporated into the OCR correction system itself so comparing system output to system input will happen automatically and running this script separately won't be necessary.
