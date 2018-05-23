#!/usr/bin/env python3
'''

@author genevp, adapted from Jimmy Bruno
'''
import argparse
from collections import OrderedDict
from itertools import chain
from os import path
import re
import sys

# used in StatsTuple:
from builtins import property as _property, tuple as _tuple
from operator import itemgetter as _itemgetter


def get_distance_matrix(ref, hypothesis):

    '''
    return an edit distance matrix
    Parameters:
    -----------
        ref : iterable
            the "reference" iterable, e.g. elements present in ref but absent
            in hypothesis will be deletions.
        hypothesis : iterable
            the "hypothesis iterable", e.g. elements present in hypothesis but
            absent in ref will be insertions
    Returns:
    --------
        distance_matrix : 2d list of lists
    '''
    # initialize the matrix
    ref_len = len(ref) + 1
    hyp_len = len(hypothesis) + 1
    distance_matrix = [[0] * hyp_len for _ in range(ref_len)]
    for i in range(ref_len):
        distance_matrix[i][0] = i
    for j in range(hyp_len):
        distance_matrix[0][j] = j

    # calculate the edit distances
    for i in range(1, ref_len):
        for j in range(1, hyp_len):

            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1]

            if ref[i-1] != hypothesis[j-1]:
                substitution += 1

            distance_matrix[i][j] = min(insertion, deletion, substitution)

    return distance_matrix


class StatsTuple(tuple):
    '''
    tuple subclass used to track edit_distance and related values.
    Copy-and-paste-and-modify of NamedTuple _source with
    addition operator overridden
    Attributes:
    ----------
        edit_distance : int
        num_deletions : int
        num_insertions :int
        num_substituions : int
        num_ref_elements :int
    '''

    __slots__ = ()

    _fields = ('edit_distance', 'num_deletions', 'num_insertions',
               'num_substituions', 'num_ref_elements')

    def __new__(_cls, edit_distance, num_deletions, num_insertions,
                num_substituions, num_ref_elements):
        '''Create new instance of DiffStats(edit_distance, num_deletions,
           num_insertions, num_substituions, num_ref_elements, alignment)'''
        return _tuple.__new__(_cls, (edit_distance, num_deletions,
                                     num_insertions, num_substituions,
                                     num_ref_elements))

    @classmethod
    def _make(cls, iterable, new=tuple.__new__, len=len):
        'Make a new DiffStats object from a sequence or iterable'
        result = new(cls, iterable)
        if len(result) != 5:
            raise TypeError('Expected 5 arguments, got %d' % len(result))
        return result

    def _replace(_self, **kwds):
        '''Return a new StatsTuple object
            replacing specified fields with new values'''
        result = _self._make(map(kwds.pop, ('edit_distance', 'num_deletions',
                                            'num_insertions',
                                            'num_substituions',
                                            'num_ref_elements'), _self))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % list(kwds))
        return result

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + (
                    '(edit_distance=%r, num_deletions=%r, num_insertions=%r, '
                    'num_substituions=%r, num_ref_elements=%r)' % self)

    def _asdict(self):
        'Return a new OrderedDict which maps field names to their values.'
        return OrderedDict(zip(self._fields, self))

    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return tuple(self)

    def __add__(self, other):
        '''
        add all the attributes together and return a new StatsTuple object
        '''
        return StatsTuple(*(i + j for i, j in zip(self, other)))

    edit_distance = _property(_itemgetter(0), doc='Alias for field number 0')
    num_deletions = _property(_itemgetter(1), doc='Alias for field number 1')
    num_insertions = _property(_itemgetter(2), doc='Alias for field number 2')
    num_substituions = _property(_itemgetter(3), doc='Alias for field number 3')
    num_ref_elements = _property(_itemgetter(4), doc='Alias for field number 4')


def get_breakpoints(elements, max_characters):
    '''
    return the indices of the element in elements to break on, so that the
    printed output does not exceed max_characters when joining the elements
    to form an putput string. Called by WERCalculator.print_alignment().
    Parameters:
    -----------
        elements : iterable
        max_characters : int
    Returns:
        breakpoints : list
    '''
    breakpoints = []

    length_tracker = 0

    for i, element in enumerate(elements):
        # We add + 1 throughout for the space when we print
        if length_tracker + len(element) + 1 > max_characters:
            # this will start a new line, so capture the index as a breakpoint
            breakpoints.append(i)
            length_tracker = len(element) + 1

        else:
            length_tracker += len(element) + 1

    return breakpoints


class WERCalculator():
    '''
    Word-Error-Rate Calculator
    Parameters:
    -----------
        ref : iterable
            the "reference" iterable, e.g. elements present in reference
            but absent in hypothesis will be deletions.
        hypothesis : iterable
            the "hypothesis" iterable, e.g. elements present in hypothesis
            but absent in reference will be insertions
    '''
    def __init__(self, reference, hypothesis):
        self.reference = reference
        self.hypothesis = hypothesis

        self.distance_matrix = get_distance_matrix(reference, hypothesis)
        i = len(self.distance_matrix) - 1
        j = len(self.distance_matrix[i]) - 1
        self.edit_distance = self.distance_matrix[i][j]
        self.num_ref_elements = i

    def __repr__(self):
        hypothesis_str = str(self.hypothesis)
        reference_str = str(self.reference)
        if len(hypothesis_str) > 10:
            hypothesis_str = hypothesis_str[10:] + " ..."

        if len(reference_str) > 10:
            reference_str = reference_str[10:] + " ..."
        return "WERCalculator({}, {})".format(hypothesis_str, reference_str)

    def wer(self):
        '''
        return the word-error-rate
        '''
        return self.edit_distance/self.num_ref_elements

    def set_diff_stats(self, prepare_alignment=False):
        '''
        set diff_stats tuple of (edit_distance, num_deletions, num_insertions,
                                 num_substituions, num_ref_elements)
        if prepare_alignment is true, then we also get the three lists we need
        to be able to print out a nice alignment (we only do this if we need
        to, because it can slow things down if the text is long)
        '''
        reference = self.reference
        hypothesis = self.hypothesis
        num_ref_elements = self.num_ref_elements
        i = num_ref_elements
        j = len(self.hypothesis)

        edit_distance = self.edit_distance
        distance_matrix = self.distance_matrix

        num_deletions = 0
        num_insertions = 0
        num_substituions = 0

        if prepare_alignment:
            # we'll need these if we want the alignment
            align_ref_elements = []
            align_hypothesis_elements = []
            align_label_str = []

        # start at the cell containing the edit distance and analyze the
        # matrix to figure out what is a deletion, insertion, or
        # substitution.
        while i or j:
            # if deletion
            if distance_matrix[i][j] == distance_matrix[i-1][j] + 1:
                num_deletions += 1

                if prepare_alignment:
                    align_ref_elements.append(reference[i-1])
                    align_hypothesis_elements.append(" ")
                    align_label_str.append('D')

                i -= 1

            # if insertion
            elif distance_matrix[i][j] == distance_matrix[i][j-1] + 1:
                num_insertions += 1

                if prepare_alignment:
                    align_ref_elements.append(" ")
                    align_hypothesis_elements.append(hypothesis[j-1])
                    align_label_str.append('I')

                j -= 1

            # if match or substitution
            else:
                ref_element = reference[i-1]
                hypothesis_element = hypothesis[j-1]

                if ref_element != hypothesis_element:
                    num_substituions += 1
                    label = 'S'
                else:
                    label = ' '

                if prepare_alignment:
                    align_ref_elements.append(ref_element)
                    align_hypothesis_elements.append(hypothesis_element)
                    align_label_str.append(label)

                i -= 1
                j -= 1

        if prepare_alignment:
            align_ref_elements.reverse()
            align_hypothesis_elements.reverse()
            align_label_str.reverse()

            self.align_ref_elements = align_ref_elements
            self.align_hypothesis_elements = align_hypothesis_elements
            self.align_label_str = align_label_str

        diff_stats = StatsTuple(edit_distance, num_deletions, num_insertions,
                                num_substituions, num_ref_elements)
        self._diff_stats = diff_stats

    @property
    def diff_stats(self):
        '''
        return the diff_stats object with edit distance and other stats
        '''
        if not hasattr(self, '_diff_stats'):
            self.set_diff_stats()
        return self._diff_stats


    def changed(self, plain_text):
        '''
        returns the plain_text string plus markup for bold and red font formatting (indicating a change)
        :param plain_text: text to change
        :return: text with markup tags
        '''

        return r"\b \cf3 " + plain_text + r" \b0 \cf2"


    def gp_show_changes(self):
        '''
        Returns the hypothesis text only, with differences between it and the reference surrounded by formatting markup
        '''

        # GP EDITS
        # treating 'ref' as correction system INPUT (top line, original), 'hyp' as OUTPUT (bottom line, changed)

        if not hasattr(self, 'align_ref_elements'):
            self.set_diff_stats(prepare_alignment=True)

        assert (len(self.align_ref_elements) ==
                len(self.align_hypothesis_elements) ==
                len(self.align_label_str))

        assert len(self.align_label_str) == len(self.align_hypothesis_elements) == len(self.align_ref_elements), "different number of elements"

        # for each word in line, determine whether there's a change and append with the according format
        print_string = ''
        for index in range(len(self.align_label_str)):
            if self.align_label_str[index] == ' ':
                print_string += self.align_hypothesis_elements[index] + ' '
            elif self.align_label_str[index] == 'S' or self.align_label_str[index] == 'I':
                element = self.align_hypothesis_elements[index]
                print_string += self.changed(element) + ' '
            else:  # a deletion - need to print what was in the original that got deleted
                element = self.align_ref_elements[index]
                print_string += self.changed('[' + element + ']')
        return print_string


def process_single_pair(args):
    '''
    process a single pair of files.  Called by main when running in single pair
    mode, or by process_batch when running in batch mode.

    Do alignment on files line by line, so alignment is only at the line level and printing
    (WERcalculator.gp_show_changes) also happens at each line
    '''

    # get non-empty lines from reference and hypothesis files
    with open(args.reference_file) as f:
        reference_lines = f.readlines()
    reference_lines = [l for l in reference_lines if l != '\n']

    # with open(args.hypothesis_file) as f:
    #     hypothesis_lines = f.readlines()
    # READ FROM STDIN FOR EDITED VERSION OF TEXT
    hypothesis_lines = [l for l in sys.stdin.readlines() if l != '\n']

    assert len(reference_lines) == len(hypothesis_lines), "Files contain different numbers of lines"

    # print header for rich text format; need to do this outside of the foreach loop for lines in the files
    header = r"{\rtf1\ansi\ansicpg1252\cocoartf1404\cocoasubrtf470{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}" \
             r"{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red255\green0\blue0;}" \
             r"\margl1440\margr1440\vieww11940\viewh7800\viewkind0\pard\tx720\tx1440\tx2160\tx2880\tx3600" \
             r"\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0\f0\fs24 \cf2"

    print(header)

    for i in range(len(reference_lines)):
        reference_words = reference_lines[i].split()
        hypothesis_words = hypothesis_lines[i].split()
        wer_calculator = WERCalculator(reference_words, hypothesis_words)

        wer_calculator.set_diff_stats(prepare_alignment=True)

        formatted_line = wer_calculator.gp_show_changes()
        # also print original line to the right if show_original flag is present
        # should probably make this into a separate method at some point
        if args.show_original:
            original_line = reference_lines[i]
            # find number of characters contained in the formatting tags so we can adjust the padding
            tags_length = 0
            for token in formatted_line.split():
                if token.startswith('\\'):
                    tags_length += len(token) + 1  # plus 1 for following space character
            padding = 100 + tags_length  # this still doesn't give nice columns, not sure how to fix
            formatted_line = "{:{padding}}{}".format(formatted_line, original_line, padding=padding)
        print(formatted_line)
        print("\\")

    print('}')


def process_batch(args):
    '''
    process a batch of files, calling process_single_pair on each one of them
    GP EDITS: deleted a bunch of stuff that's not relevant given my other changes but haven't tested that
    this method still works.
    Also there might still be extraneous lines.
    '''
    running_total_diff_stats = StatsTuple(0, 0, 0, 0, 0)

    line_counter = 0

    with open(args.mapping_file) as f:

        for line in f.readlines():

            line_counter += 1

            parsed_line = line.split()

            if not parsed_line:
                continue

            if parsed_line[0].startswith("#"):
                continue

            if len(parsed_line) != 2:
                print("Error: line {} of mapping file contains more than a "
                      "pair of paths".format(i))
                continue

            # this is a little hacky, manipulating the args object,
            # but it seems to work
            args.reference_file, args.hypothesis_file = parsed_line

            try:
                temp_diff_stats = process_single_pair(args)
            except FileNotFoundError as e:
                print("[Errno {}] processing line {} of {}: No such file: {}".format(
                        e.errno, line_counter, args.mapping_file, e.filename))
                continue

            running_total_diff_stats += temp_diff_stats


def main():
    # set up the main parser
    parser = argparse.ArgumentParser(
        description="Compare an original text and an edited text "
                    "and output the changes in the edited text in bolded red font.")

    # set up subparser for single pair mode
    subparsers = parser.add_subparsers(
                        title='subcommands',
                        help='indicates batch processing or single pair mode')

    single_parser = subparsers.add_parser('single',
                                          help='compare a single pair of edited and original texts')
    # main function for this sub_parser:
    single_parser.set_defaults(main_func=process_single_pair)

    # arguments
    single_parser.add_argument("reference_file",
                               help='File to use as original')
    # READING FROM STDIN INSTEAD
    # single_parser.add_argument("hypothesis_file",
    #                            help='File to use as edited')
    single_parser.add_argument('--show_original',
                               help='displays the original and edited texts side by side for easier comparison',
                               action='store_true',
                               default=False)

     # set up subparser for batch mode
    batch_parser = subparsers.add_parser(
                    'batch',
                    help='compare a batch of pairs of original and '
                         'edited texts, stored in a file '
                         'indicating their mapping')

    # main function for batch mode:
    batch_parser.set_defaults(main_func=process_batch)

    # argument
    batch_parser.add_argument(
                    'mapping_file',
                    help='file of mappings of original files to their '
                         'edited version files.  Each line represents one '
                         'mapping, where the first item on the line is a path '
                         'to the original file, followed by whitespace, '
                         'followed by the path to the edited file', )

    args = parser.parse_args()
    args.main_func(args)

if __name__ == '__main__':
    main()