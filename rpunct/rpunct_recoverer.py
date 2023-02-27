# Copyright (c) 2022 British Broadcasting Corporation

"""
Module supporting punctuation recovery and post-processing of raw STT output.
"""
import os
import re
import torch
import decimal
from jiwer import wer
from num2words import num2words

from punctuate import RestorePuncts
from number_recoverer import NumberRecoverer

class RPunctRecoverer:
    """
    A class for loading the RPunct object and exposing it to linguine code.
    """
    def __init__(self, model_source):
            self.recoverer = RestorePuncts(
                model_source=model_source,
                use_cuda=torch.cuda.is_available()
            )
            self.number_recoverer = NumberRecoverer(
                wordify_large_numbers=True,
                correct_currencies=True,
                correct_bbc_style=True,
                correct_commas=True
            )

    def strip_punctuation(self, truth_text):
        """
        Converts a string of truth text to plaintext of the same format as STT transcripts.
        """
        # set lowercase and replace certain characters
        text = truth_text.lower()
        text = text.replace("\n", " ")
        text = text.replace(" - ", " ")
        text = text.replace("' ", " ")
        text = text.replace("%", " percent")

        # convert to list
        text = text.split(" ")
        plaintext = []

        for word in text:
            # if numerical, convert to a word
            try:
                word = num2words(word)
            except decimal.InvalidOperation:
                pass

            plaintext.append(word)

        plaintext = " ".join(plaintext)

        plaintext = plaintext.replace("-", "- ")
        plaintext = re.sub(r"[^0-9a-zA-Z' ]", "", plaintext)

        return plaintext

    def recover(self, transcript):
        """
        RPunct processes the entire block of text at once (but internally splits it into segments due to BERT
        only being able to handle a certain input length)
        This function flattens list of segments into a single block of text, then passes to RPunct

        It then reconstructs the original segments from the punctuated output, while
        applying additional capitalisation/full stops to the beginning and end of a segment, if not present

        Args:
            list_of_segs: a list of lists containing Item objects.

        Returns:
            A list of of lists containing Item objects (where each Item has added punctuation).
        """
        # Process entire transcript, then retroactively apply punctuation to words in segments
        recovered = self.recoverer.punctuate(transcript, lang='en')

        # Revert numbers to digit notation
        recovered = self.number_recoverer.process(recovered)

        return recovered

    def word_error_rate(self, truth, stripped, predicted):
        wer_plaintext = wer(truth, stripped) * 100
        word_error_rate = wer(truth, predicted) * 100
        print("\nWord error rate:")
        print(f"\tNo recovery     : {wer_plaintext:.2f}%")
        print(f"\tRPunct recovery : {word_error_rate:.2f}%", end='\n\n')

    def run(self, input_path, output_path=None, clean_up_input=True, compute_wer=True):
        # Read input text
        print(f"\nReading plaintext from file: {input_path}")
        with open(input_path, 'r') as fp:
            input_text = fp.read()

        # Convert truth transcript to plaintext (no punctuation)
        if clean_up_input:
            plaintext = self.strip_punctuation(input_text)
        else:
            plaintext = input_text

        # Use RPunct to punctuate plaintext
        punctuated = self.recover(plaintext)

        print("\nInput:", input_text, end='\n\n')
        print("Plaintext:", plaintext, end='\n\n')

        # Write output to TXT file or to the terminal
        if output_path is None:
            # print output to command line
            print("\nPrinting punctuated text", end='\n\n')
            print(punctuated)
        else:
            # Check if output directory exists
            output_dir, _ = os.path.split(output_path)
            output_path_exists = os.path.isdir(output_dir)

            # print punctuated text to output file
            if output_path_exists:
                print(f"Writing punctuated text to file: {output_path}")
                with open(output_path, 'w') as fp:
                    fp.write(punctuated)
            else:
                raise FileNotFoundError(f"Directory specified to ouptut text file to does not exist: {output_dir}")

        # Compute WER metric
        if compute_wer:
            self.word_error_rate(input_text, plaintext, punctuated)


        # USING THIS TO FIND ERROR IN LENGTH DIFFERENCES
        # Comparing transcript lengths
        print(f"> Original transcript length = {len(plaintext.split(' '))}")
        print(f"> Restored transcript length = {len(punctuated.split(' '))}")
        print(f"\t * Hyphens added = words concatenated = {punctuated.count('-')}")
        print(f"\t * Currency symbols added = keywords removed = {punctuated.count('£') + punctuated.count('$') + punctuated.count('€')}")
        print(f"\t * Deminals added = point words removed = {plaintext.count(' point ')}")


def main(topic='full-ep', model='composite20_2e'):
    # parameters
    model_location = f'outputs/{model}'
    input = f'tests/inferences/{topic}/truth.txt'
    output = f'tests/inferences/{topic}/{model}.txt'
    output = None

    input = 'tests/inferences/competitor_comparisons/testdotwav_rpunct.txt'
    output = 'tests/inferences/competitor_comparisons/testdotwav_plain.txt'

    # generate instance of rpunct model
    punct_model = RPunctRecoverer(model_source=model_location)

    # run e2e pipeline
    punct_model.run(input, output)


if __name__ == "__main__":
    main()
