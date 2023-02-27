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

try:
    from rpunct.punctuate import RestorePuncts
    from rpunct.number_recoverer import NumberRecoverer
except ModuleNotFoundError:
    from punctuate import RestorePuncts
    from number_recoverer import NumberRecoverer

class RPunctRecoverer:
    """
    A class for loading the RPunct object and exposing it to linguine code.
    """
    def __init__(self, model_source, use_cuda):
            self.recoverer = RestorePuncts(
                model_source=model_source,
                use_cuda=(use_cuda and torch.cuda.is_available())
            )
            self.number_recoverer = NumberRecoverer()

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
        # Process entire transcript, then retroactively apply punctuation to words in segments
        recovered = self.recoverer.punctuate(transcript, lang='en')

        # Revert numbers to digit notation
        recovered = self.number_recoverer.process(recovered)

        return recovered

    def word_error_rate(self, truth, stripped, predicted):
        # Uses `jiwer` to compute word error rates between punctuated and unpunctuated text
        wer_plaintext = wer(truth, stripped) * 100
        word_error_rate = wer(truth, predicted) * 100
        print("Word error rate:")
        print(f"\tNo recovery     : {wer_plaintext:.2f}%")
        print(f"\tRPunct recovery : {word_error_rate:.2f}%", end='\n\n')

    def run(self, input_path, output_file_path=None, clean_up_input=True, compute_wer=False):
        # Read input text
        print(f"\nReading input text from file: {input_path}")
        with open(input_path, 'r') as fp:
            input_text = fp.read()

        # Convert input transcript to plaintext (no punctuation)
        if clean_up_input:
            plaintext = self.strip_punctuation(input_text)
        else:
            plaintext = input_text

        # Restore punctuation to plaintext using RPunct
        punctuated = self.recover(plaintext)

        # Output restored text (to a specified TXT file or the command line)
        if not output_file_path:
            # Output to command line
            print("\nPrinting punctuated text:", end='\n\n')
            print(punctuated, end='\n\n')
        else:
            # Check if output directory exists
            output_dir, _ = os.path.split(output_file_path)
            output_path_exists = os.path.isdir(output_dir)

            # Output to file if the directory exists
            if output_path_exists:
                print(f"Writing punctuated text to file: {output_file_path}")
                with open(output_file_path, 'w') as fp:
                    fp.write(punctuated)
            else:
                raise FileNotFoundError(f"Directory specified to ouptut text file to does not exist: {output_dir}")

        # Compute WER metric
        if compute_wer:
            self.word_error_rate(input_text, plaintext, punctuated)

        # # USING THIS TO FIND ERROR IN LENGTH DIFFERENCES
        # # Comparing transcript lengths
        # print(f"> Original transcript length = {len(plaintext.split(' '))}")
        # print(f"> Restored transcript length = {len(punctuated.split(' '))}")
        # print(f"\t * Hyphens added = words concatenated = {punctuated.count('-')}")
        # print(f"\t * Currency symbols added = keywords removed = {punctuated.count('£') + punctuated.count('$') + punctuated.count('€')}")
        # print(f"\t * Deminals added = point words removed = {plaintext.count(' point ')}")


def rpunct_main(model_location, input_txt, output_txt=None, use_cuda=False):
    # Generate an RPunct model instance
    punct_model = RPunctRecoverer(model_source=model_location, use_cuda=use_cuda)

    # Run e2e inference pipeline
    punct_model.run(input_txt, output_txt)


if __name__ == "__main__":
    model_default = 'outputs/clean-composite-1e'
    input_default = 'tests/inferences/full-ep/truth.txt'
    rpunct_main(model_default, input_default)
