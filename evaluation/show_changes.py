#!/usr/bin/env python3
'''
Compares an edited text to the original line by line and prints the edited text with changes in bold, red font.
Deletions are enclosed in square brackets.
--show_original option prints the edited and original texts side by side
--numbered prints line numbers on the left margin

@author genevp, adapted from Jimmy Bruno's wer.py
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


def get_distance_matrix(orig, edited):

    '''
    return an edit distance matrix
    Parameters:
    -----------
        orig : iterable
            the "original" iterable, e.g. elements present in orig but absent
            in edited will be deletions.
        edited : iterable
            the "edited iterable", e.g. elements present in edited but
            absent in orig will be insertions
    Returns:
    --------
        distance_matrix : 2d list of lists
    '''

    # initialize the matrix
    orig_len = len(orig) + 1
    edit_len = len(edited) + 1
    distance_matrix = [[0] * edit_len for _ in range(orig_len)]
    for i in range(orig_len):
        distance_matrix[i][0] = i
    for j in range(edit_len):
        distance_matrix[0][j] = j

    # calculate the edit distances
    for i in range(1, orig_len):
        for j in range(1, edit_len):

            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1]

            if orig[i-1] != edited[j-1]:
                substitution += 1

            distance_matrix[i][j] = min(insertion, deletion, substitution)

    return distance_matrix


# class StatsTuple(tuple):
#     '''
#     tuple subclass used to track edit_distance and related values.
#     Copy-and-paste-and-modify of NamedTuple _source with
#     addition operator overridden
#     Attributes:
#     ----------
#         edit_distance : int
#         num_deletions : int
#         num_insertions :int
#         num_substitutions : int
#         num_orig_elements :int
#     '''
#
#     __slots__ = ()
#
#     _fields = ('edit_distance', 'num_deletions', 'num_insertions',
#                'num_substitutions', 'num_orig_elements')
#
#     def __new__(_cls, edit_distance, num_deletions, num_insertions,
#                 num_substitutions, num_orig_elements):
#         '''Create new instance of DiffStats(edit_distance, num_deletions,
#            num_insertions, num_substitutions, num_orig_elements, alignment)'''
#         return _tuple.__new__(_cls, (edit_distance, num_deletions,
#                                      num_insertions, num_substitutions,
#                                      num_orig_elements))
#
#     @classmethod
#     def _make(cls, iterable, new=tuple.__new__, len=len):
#         'Make a new DiffStats object from a sequence or iterable'
#         result = new(cls, iterable)
#         if len(result) != 5:
#             raise TypeError('Expected 5 arguments, got %d' % len(result))
#         return result
#
#     def _replace(_self, **kwds):
#         '''Return a new StatsTuple object
#             replacing specified fields with new values'''
#         result = _self._make(map(kwds.pop, ('edit_distance', 'num_deletions',
#                                             'num_insertions',
#                                             'num_substitutions',
#                                             'num_orig_elements'), _self))
#         if kwds:
#             raise ValueError('Got unexpected field names: %r' % list(kwds))
#         return result
#
#     def __repr__(self):
#         'Return a nicely formatted representation string'
#         return self.__class__.__name__ + (
#                     '(edit_distance=%r, num_deletions=%r, num_insertions=%r, '
#                     'num_substitutions=%r, num_orig_elements=%r)' % self)
#
#     def _asdict(self):
#         'Return a new OrderedDict which maps field names to their values.'
#         return OrderedDict(zip(self._fields, self))
#
#     def __getnewargs__(self):
#         'Return self as a plain tuple.  Used by copy and pickle.'
#         return tuple(self)
#
#     def __add__(self, other):
#         '''
#         add all the attributes together and return a new StatsTuple object
#         '''
#         return StatsTuple(*(i + j for i, j in zip(self, other)))
#
#     edit_distance = _property(_itemgetter(0), doc='Alias for field number 0')
#     num_deletions = _property(_itemgetter(1), doc='Alias for field number 1')
#     num_insertions = _property(_itemgetter(2), doc='Alias for field number 2')
#     num_substitutions = _property(_itemgetter(3), doc='Alias for field number 3')
#     num_orig_elements = _property(_itemgetter(4), doc='Alias for field number 4')
#

# def get_breakpoints(elements, max_characters):
#     '''
#     return the indices of the element in elements to break on, so that the
#     printed output does not exceed max_characters when joining the elements
#     to form an putput string. Called by WERCalculator.print_alignment().
#     Parameters:
#     -----------
#         elements : iterable
#         max_characters : int
#     Returns:
#         breakpoints : list
#     '''
#     breakpoints = []
#
#     length_tracker = 0
#
#     for i, element in enumerate(elements):
#         # We add + 1 throughout for the space when we print
#         if length_tracker + len(element) + 1 > max_characters:
#             # this will start a new line, so capture the index as a breakpoint
#             breakpoints.append(i)
#             length_tracker = len(element) + 1
#
#         else:
#             length_tracker += len(element) + 1
#
#     return breakpoints


class Compare:
    '''
    
    Parameters:
    -----------
        orig : iterable
            the "original" iterable, e.g. elements present in origerence
            but absent in edited will be deletions.
        edited : iterable
            the "edited" iterable, e.g. elements present in edited
            but absent in original will be insertions
    '''
    def __init__(self, original, edited):
        self.original = original
        self.edited = edited

        self.distance_matrix = get_distance_matrix(original, edited)
        i = len(self.distance_matrix) - 1
        j = len(self.distance_matrix[i]) - 1
        self.edit_distance = self.distance_matrix[i][j]
        self.num_orig_elements = i

    def __repr__(self):
        edited_str = str(self.edited)
        original_str = str(self.original)
        if len(edited_str) > 10:
            edited_str = edited_str[10:] + " ..."

        if len(original_str) > 10:
            original_str = original_str[10:] + " ..."
        return "Compare({}, {})".format(edited_str, original_str)

    def set_alignment_strings(self):
        '''
        get aligned [corresponding] elements of original, edited, and labels and set Compare object attributes
        '''

        original = self.original
        edited = self.edited
        num_orig_elements = self.num_orig_elements
        i = num_orig_elements
        j = len(self.edited)

        # edit_distance = self.edit_distance
        distance_matrix = self.distance_matrix

        num_deletions = 0
        num_insertions = 0
        num_substitutions = 0

        align_orig_elements = []
        align_edited_elements = []
        align_label_str = []

        # start at the cell containing the edit distance and analyze the
        # matrix to figure out what is a deletion, insertion, or
        # substitution.
        while i or j:
            # if deletion
            if distance_matrix[i][j] == distance_matrix[i-1][j] + 1:
                num_deletions += 1

                align_orig_elements.append(original[i-1])
                align_edited_elements.append(" ")
                align_label_str.append('D')

                i -= 1

            # if insertion
            elif distance_matrix[i][j] == distance_matrix[i][j-1] + 1:
                num_insertions += 1

                align_orig_elements.append(" ")
                align_edited_elements.append(edited[j-1])
                align_label_str.append('I')

                j -= 1

            # if match or substitution
            else:
                orig_element = original[i-1]
                edited_element = edited[j-1]

                if orig_element != edited_element:
                    num_substitutions += 1
                    label = 'S'
                else:
                    label = ' '

                align_orig_elements.append(orig_element)
                align_edited_elements.append(edited_element)
                align_label_str.append(label)

                i -= 1
                j -= 1

        align_orig_elements.reverse()
        align_edited_elements.reverse()
        align_label_str.reverse()

        self.align_orig_elements = align_orig_elements
        self.align_edited_elements = align_edited_elements
        self.align_label_str = align_label_str

    def show_changes(self):
        '''
        Returns the edited text only, with differences between it and the original surrounded by formatting markup
        'orig' is correction system INPUT (top line), 'edit' is OUTPUT (bottom line)

        '''

        if not hasattr(self, 'align_orig_elements'):
            self.set_alignment_strings()

        assert (len(self.align_orig_elements) ==
                len(self.align_edited_elements) ==
                len(self.align_label_str))

        assert len(self.align_label_str) == len(self.align_edited_elements) == len(self.align_orig_elements), "different number of elements"

        # for each word in line, determine whether there's a change and append with the according format
        print_string = ''
        for index in range(len(self.align_label_str)):
            if self.align_label_str[index] == ' ':
                print_string += self.align_edited_elements[index] + ' '
            elif self.align_label_str[index] == 'S' or self.align_label_str[index] == 'I':
                element = self.align_edited_elements[index]
                print_string += changed(element) + ' '
            else:  # a deletion - need to print what was in the original that got deleted
                element = self.align_orig_elements[index]
                print_string += changed('[' + element + ']')
        return print_string


def changed(plain_text):
    '''
    returns plain_text surrounded by markup tags for bold, red font formatting (indicating a change from the original)
    :param plain_text: text to change
    :return: text with markup tags
    '''

    return r"\b \cf3 " + plain_text + r" \b0 \cf2"


def process_single_pair(args):
    '''
    process a single pair of files.  Called by main when running in single pair
    mode, or by process_batch when running in batch mode.

    Do alignment on files line by line, so alignment is only at the line level and printing
    (Compare.show_changes) also happens at each line
    '''

    # get non-empty lines from original and edited files
    with open(args.original_file) as f:
        original_lines = f.readlines()
    original_lines = [line for line in original_lines if line != '\n']

    # READ EDITED TEXT FROM STDIN
    edited_lines = [line for line in sys.stdin.readlines() if line != '\n']

    assert len(original_lines) == len(edited_lines), "Files contain different numbers of lines"

    # print header for rich text format; need to do this outside of the foreach loop for lines in the files
    header = r"{\rtf1\ansi\ansicpg1252\cocoartf1404\cocoasubrtf470{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}" \
             r"{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red255\green0\blue0;}" \
             r"\margl1440\margr1440\vieww11940\viewh7800\viewkind0\pard\tx720\tx1440\tx2160\tx2880\tx3600" \
             r"\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0\f0\fs24 \cf2"

    print(header)

    for i in range(len(original_lines)):
        original_words = original_lines[i].split()
        edited_words = edited_lines[i].split()
        comparison = Compare(original_words, edited_words)

        comparison.set_alignment_strings()

        formatted_line = comparison.show_changes()

        # add line numbers if --numbered flag is present
        if args.numbered:
            formatted_line = str(i + 1) + ' ' + formatted_line

        # also print original line to the right if show_original flag is present
        # (should probably make this into a separate method at some point)
        if args.show_original:
            original_line = original_lines[i]
            # find number of characters contained in the formatting tags so we can adjust the padding
            tags_length = 0
            for token in formatted_line.split():
                if token.startswith('\\'):
                    tags_length += len(token) + 1  # plus 1 for following space character
            padding = 100 + tags_length
            formatted_line = "{:{padding}}{}".format(formatted_line, original_line, padding=padding)

        print(formatted_line)
        print("\\")

    print('}')


def main():
    # set up the main parser
    parser = argparse.ArgumentParser(
        description="Compare an original text and an edited text "
                    "and output the edited text with the differences in bolded red font.")

    # main function for this parser:
    parser.set_defaults(main_func=process_single_pair)

    # arguments
    parser.add_argument("original_file",
                        help='File to use as original')

    parser.add_argument('--show_original',
                        help='display the original and edited texts side by side for easier comparison',
                        action='store_true',
                        default=False)
    parser.add_argument('--numbered',
                        help='display line numbers',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    args.main_func(args)


if __name__ == '__main__':
    main()