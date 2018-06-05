A program to show where an edited text (such as the output from an OCR correction system) differs from the original (such as the raw OCR'd text). 

# Input
The original text file, in plain text format (.txt), and the edited text passed through standard input. It is assumed that the texts have line-by-line correspondence (line breaks are the same).

# Output 
The edited text (in .rtf format) with changes in bold and red. Words that appear in the original text but not in the edited text appear inside square brackets: "\[\]" (see example output below). The rich text file format should be openable by your operating system's basic text editor or in Google Drive. 

> __[:]__ The boisterous Mor Whymper came to __breakfast. .....__ I've decided all that 
> excitable volubility in his company __B......__ er because when he crosses over 
> with Don's and me in the felucca on our way to church he is __Euite__ pleasantly 

To run from the command line: 
* Make sure you have python installed. (You can check by opening your command line utility (Terminal for Macs, Command Prompt for Windows) and typing "python".) 
* Navigate to the directory that contains the show_changes.py file.
* Run the following command:

```
python show_changes.py single path/original_file.txt path/output_file.rtf < edited_text 
```

## NOTE
This process is intended to be the last step of a pipeline that starts with the OCR correction system itself, not a stand-alone component, so running it individually like this should be uncommon.