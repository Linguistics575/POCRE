#!/usr/bin/env python3

'''
Script to get a printed alignment between reference and hypothesis texts.
Horizontally printed alignment looks like this:
==== Fuzzy Wuzzy was a==== bear
John Fuzzy Wuzzy had hair. ====

Vertically printed alignment looks like this:
===== John
Fuzzy Fuzzy
Wuzzy Wuzzy
was   had
a==== hair.
bear  ====

Adapted by GP from @author Jimmy Bruno's WER.py

GP EDITS:
added '=' (as null character) for padding instead of default ' ' in horizontal alignment output
removed labels in alignment output
removed error rate statistics in output
removed -verbose option
removed -a argument so that outputting alignment is default behavior
'''

import argparse
from collections import OrderedDict
from itertools import chain
from os import path

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

            deletion = distance_matrix[i - 1][j] + 1
            insertion = distance_matrix[i][j - 1] + 1
            substitution = distance_matrix[i - 1][j - 1]

            if ref[i - 1] != hypothesis[j - 1]:
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
        num_substitutions : int
        num_ref_elements :int
    '''

    __slots__ = ()

    _fields = ('edit_distance', 'num_deletions', 'num_insertions',
               'num_substitutions', 'num_ref_elements')

    def __new__(_cls, edit_distance, num_deletions, num_insertions,
                num_substitutions, num_ref_elements):
        '''Create new instance of DiffStats(edit_distance, num_deletions,
           num_insertions, num_substitutions, num_ref_elements, alignment)'''
        return _tuple.__new__(_cls, (edit_distance, num_deletions,
                                     num_insertions, num_substitutions,
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
                                            'num_substitutions',
                                            'num_ref_elements'), _self))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % list(kwds))
        return result

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + (
            '(edit_distance=%r, num_deletions=%r, num_insertions=%r, '
            'num_substitutions=%r, num_ref_elements=%r)' % self)

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
    num_substitutions = _property(_itemgetter(3), doc='Alias for field number 3')
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

    def set_diff_stats(self, prepare_alignment=False):
        '''
        set diff_stats tuple of (edit_distance, num_deletions, num_insertions,
                                 num_substitutions, num_ref_elements)
        we also get the two lists we need for the alignment
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
        num_substitutions = 0

        if prepare_alignment:
            # we'll need these if we want the alignment
            align_ref_elements = []
            align_hypothesis_elements = []

        # start at the cell containing the edit distance and analyze the
        # matrix to figure out what is a deletion, insertion, or
        # substitution.
        while i or j:
            # if deletion
            if distance_matrix[i][j] == distance_matrix[i - 1][j] + 1:
                num_deletions += 1

                if prepare_alignment:
                    align_ref_elements.append(reference[i - 1])
                    align_hypothesis_elements.append(" ")

                i -= 1

            # if insertion
            elif distance_matrix[i][j] == distance_matrix[i][j - 1] + 1:
                num_insertions += 1

                if prepare_alignment:
                    align_ref_elements.append(" ")
                    align_hypothesis_elements.append(hypothesis[j - 1])

                j -= 1

            # if match or substitution
            else:
                ref_element = reference[i - 1]
                hypothesis_element = hypothesis[j - 1]

                if ref_element != hypothesis_element:
                    num_substitutions += 1

                if prepare_alignment:
                    align_ref_elements.append(ref_element)
                    align_hypothesis_elements.append(hypothesis_element)

                i -= 1
                j -= 1

        if prepare_alignment:
            align_ref_elements.reverse()
            align_hypothesis_elements.reverse()

            self.align_ref_elements = align_ref_elements
            self.align_hypothesis_elements = align_hypothesis_elements

        diff_stats = StatsTuple(edit_distance, num_deletions, num_insertions,
                                num_substitutions, num_ref_elements)
        self._diff_stats = diff_stats

    @property
    def diff_stats(self):
        '''
        return the diff_stats object with edit distance and other stats
        '''
        if not hasattr(self, '_diff_stats'):
            self.set_diff_stats()
        return self._diff_stats

    def print_alignment(self, orient='horizontal'):
        '''
        pretty prints an alignment to stdout
        Parameters:
        -----------
            orient : str ('horizontal' or 'vertical', defaults to 'horizontal')
                orientation of printout. 'horizontal' will insert new lines
                at about 70 characters across.
        '''

        assert orient == 'horizontal' or orient == 'vertical'

        if not hasattr(self, 'align_ref_elements'):
            self.set_diff_stats(prepare_alignment=True)

        assert (len(self.align_ref_elements) ==
                len(self.align_hypothesis_elements))

        if orient == 'horizontal':
            # we'll need to pad things to elements line up nicely horizontally

            # list of the maximumum lengths of  (reference, hypothesis)
            max_lengths = [max(map(len, e)) for e
                           in zip(self.align_ref_elements, self.align_hypothesis_elements)]

            null_char_list = ['='] * len(
                max_lengths)  # GP EDIT (iterable of '=' to use as fill character for padding below)

            # list of reference elements with padding appropriate for printing
            padded_ref_elements = list(map(str.ljust,
                                           self.align_ref_elements,
                                           max_lengths, null_char_list))

            # list of hypothesis elements with padding appropriate for printing
            padded_hyp_elements = list(map(str.ljust,
                                           self.align_hypothesis_elements,
                                           max_lengths, null_char_list))

            # breakpoints that indicate the element that starts a new line.
            # this is so that we can cut the output off at 80 characters so it
            # doesn't run off of the screen
            breakpoints = get_breakpoints(padded_ref_elements, 79)

            start_index = 0
            end_index = None

            # print the first slice if there are any breakpoints
            if breakpoints:
                end_index = breakpoints[0]
                print(" ".join(padded_ref_elements[start_index:end_index]))
                print(" ".join(padded_hyp_elements[start_index:end_index]))
                print("")

                # iterate through the rest and print the lines
                for start_index, end_index in zip(*[breakpoints[i:]
                                                    for i in range(2)]):
                    print(" ".join(padded_ref_elements[start_index:end_index]))
                    print(" ".join(padded_hyp_elements[start_index:end_index]))
                    print("")

                # if there was 2 or more breakpoints, there will be an
                # end_index here, and that will be the start_index for the last
                # printing
                if end_index:
                    start_index = end_index

            # and print the last one left in the "buffer", or perhaps the only
            # one that exists
            print(" ".join(padded_ref_elements[start_index:]))
            print(" ".join(padded_hyp_elements[start_index:]))
            print("")
        else:
            # we'll need to pad things to create nice columns, which means that
            # we just have to add padding to the right side of the references

            # maximum length of any element
            max_length = max(map(len,
                                 list(chain(self.align_ref_elements,
                                            self.align_hypothesis_elements))))
            # do the padding
            padded_ref_elements = list(map(lambda x: str.ljust(x, max_length),
                                           self.align_ref_elements))
            for x in zip(
                    padded_ref_elements,
                    self.align_hypothesis_elements):
                print(" ".join(x))


def process_single_pair(args):
    '''
    process a single pair of files.  Called by main when running in single pair
    mode, or by process_batch when running in batch mode.
    '''
    with open(args.reference_file) as f:
        reference = f.read().split()

    with open(args.hypothesis_file) as f:
        hypothesis = f.read().split()

    wer_calculator = WERCalculator(reference, hypothesis)

    # prepare for printing alignment
    wer_calculator.set_diff_stats(prepare_alignment=True)
    wer_calculator.print_alignment(orient=args.vertical)


def process_batch(args):
    '''
    process a batch of files, calling process_single_pair on each one of them
    '''

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
                      "pair of paths".format(i), file=stderr)
                continue

            # this is a little hacky, manipulating the args object,
            # but it seems to work
            args.reference_file, args.hypothesis_file = parsed_line

            try:
                process_single_pair(args)
            except FileNotFoundError as e:
                print("[Errno {}] processing line {} of {}: No such file: {}".format(
                    e.errno, line_counter, args.mapping_file, e.filename),
                      file=stderr)
                continue


def main():
    # set up the main parser
    parser = argparse.ArgumentParser(
        description="Produces a word-level alignment between reference and hypothesis texts")

    # set up subparser for single pair mode
    subparsers = parser.add_subparsers(
        title='subcommands',
        help='indicates batch processing or single pair mode')

    single_parser = subparsers.add_parser('single',
                                          help='produce alignment of a single '
                                               'pair of reference and '
                                               'hypothesis files')
    # main function for this sub_parser:
    single_parser.set_defaults(main_func=process_single_pair)

    # arguments
    single_parser.add_argument("reference_file",
                               help='File to use as Reference')
    single_parser.add_argument("hypothesis_file",
                               help='File to use as Hypothesis')

    single_parser.add_argument("--vertical", help="Print alignment vertically", default='horizontal')

    # set up subparser for batch mode
    batch_parser = subparsers.add_parser(
        'batch',
        help='produce an alignment of pairs of reference '
             'files and hypothesis files, stored in a file '
             'indicating their mapping')

    # main function for batch mode:
    batch_parser.set_defaults(main_func=process_batch)

    # argument
    batch_parser.add_argument(
        'mapping_file',
        help='file of mappings of reference files to thier '
             'hypothesis files.  Each line represents one '
             'mapping, where the first item on the line is a path '
             'to the reference file, followed by whitespace, '
             'followed by the path to the hypothesis file')

    args = parser.parse_args()
    args.main_func(args)


if __name__ == '__main__':
    main()
