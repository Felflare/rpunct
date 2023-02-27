# Copyright (c) 2022 British Broadcasting Corporation

"""
Module supporting punctuation recovery and post-processing of raw STT output.
"""
import re
import string
from num2words import num2words
from number_parser import parse as number_parser

try:
    from rpunct.punctuate import TERMINALS
except ModuleNotFoundError:
    from punctuate import TERMINALS

LARGE_NUMBERS = ['million', 'billion', 'trillion']
currencies = {
    'pound': '£',
    'euro': '€',
    'dollar': '$'
}

class NumberRecoverer:
    """
    Parent class for number recovery, initially just a wrapper around number_parser
    """

    def __init__(self, wordify_large_numbers=True, correct_currencies=True, correct_bbc_style=True, correct_commas=True):
        self.wordify_large_numbers = wordify_large_numbers
        self.correct_currencies = correct_currencies
        self.correct_bbc_style = correct_bbc_style
        self.correct_commas = correct_commas

    def number_parser(self, text):
        # BBC Style Guide asserts that single digit numbers should be written as words, so revert those numbers
        # also ensure large number definitions remain as words in text by replacing with control characters
        if self.wordify_large_numbers:
            for index, number in enumerate(LARGE_NUMBERS):
                text = text.replace(number, f"\\{index}")

            # convert all other numbers to digits
            parsed = number_parser(text)

            # return control charcaters to words
            for index, number in enumerate(LARGE_NUMBERS):
                parsed = parsed.replace(f"\\{index}", number)
        else:
            # convert all other numbers to digits
            parsed = number_parser(text)

        return parsed

    def process(self, text):
        """
        Apply number recovery to a text string

        Args:
            text: String of punctuated text.

        Returns:
            A string of where numerical words are converted to numbers where applicable.
        """
        # convert numerical strings to digits in text
        parsed_text = self.number_parser(text)

        # number-parser adds spaces around numbers, so re-concatenate to any trailing punctuation
        parsed_text = parsed_text.replace(" percent", "%")
        for punct in string.punctuation:
            parsed_text = parsed_text.replace(f" {punct}", f"{punct}")

        for key in currencies.keys():  # replace any hyphenated currency words
            parsed_text = parsed_text.replace(f"-{key}", f" {key}")

        # restore decimal point notation
        parsed_list = parsed_text.split(" ")
        parsed_list = self.replace_decimal_points(parsed_list)

        # Format the style of the output text
        output_text = ""

        for word in parsed_list:
            stripped_word = re.sub(r"[^0-9a-zA-Z]", "", word).lower()

            # Restore currency words to their symbols
            if self.correct_currencies and self.is_currency(stripped_word):
                    output_text = self.insert_currency_symbols(output_text, word)

            # BBC Style Guide asserts that single digit numbers should be written as words, so revert those numbers
            elif self.correct_bbc_style and self.is_stylable(word):
                    output_text = self.bbc_style_numbers(output_text, word)

            # Format long numbers into thousands separared with commas
            elif self.correct_commas and stripped_word.isnumeric() and int(stripped_word) >= 10000:
                output_text += self.insert_number_commas(word) + " "

            else:
                output_text += word + " "

        output_text = output_text.strip()
        output_text = output_text.replace(" - ", "-")
        output_text = output_text.replace("- ", "-")

        return output_text

    def is_currency(self, word):
        if word in currencies.keys() or (word[-1] == 's' and word[:-1] in currencies.keys()):
            return True
        else:
            return False

    def is_stylable(self, word):
        if (word.isnumeric() and int(word) < 10) or (not word[-1].isnumeric() and word[:-1].isnumeric() and int(word[:-1]) < 10):
            return True
        else:
            return False

    @staticmethod
    def replace_decimal_points(text_list):
        corrected_list = []
        i = 0

        while i < len(text_list):
            word = text_list[i]

            # check if the word "point" has appeared
            if re.sub(r"[^0-9a-zA-Z]", "", word) == "point" and i > 0 and i < len(text_list) - 1:
                pre_word = text_list[i - 1]
                pre_word_stripped = re.sub(",.?!%", "", pre_word)
                post_word = text_list[i + 1]
                post_word_stripped = re.sub(r"[^0-9a-zA-Z]", "", post_word)

                if pre_word_stripped.isnumeric() and post_word_stripped.isnumeric():
                    full_number = pre_word_stripped + '.' + post_word
                    corrected_list = corrected_list[:-1]
                    corrected_list.append(full_number)
                    i += 2
                else:
                    corrected_list.append(word)
                    i += 1
            else:
                corrected_list.append(word)
                i += 1

        return corrected_list

    def insert_currency_symbols(self, text, currency):
        # get (singular) plaintext version of currency keyword
        stripped_currency = re.sub(r"[^0-9a-zA-Z]", "", currency).lower()
        if stripped_currency[-1] == 's':
            stripped_currency = stripped_currency[:-1]

        found = False
        lookback = 1

        # scan through lookback window to find a numeric word to punctuate with the currency symbol
        while lookback < 4:
            output_text_split = text.split(" ")
            prev_word = output_text_split[-lookback]
            prev_word_stripped = re.sub(r"[^0-9a-zA-Z]", "", prev_word)

            # when a numeric word is found, reconstruct the output text around this
            if prev_word_stripped.isnumeric():
                new_output_text = output_text_split[:-lookback]  # text before currency symbol
                new_output_text.append(currencies.get(stripped_currency) + prev_word)  # currency number
                new_output_text.extend(output_text_split[-lookback + 1:])  # text after currency symbol
                text = " ".join(new_output_text)

                # add any punctuation trailing the original currency keyword
                if not currency[-1].isalnum():
                    text = text[:-1] + currency[-1] + " "

                found = True
                break
            else:
                lookback += 1

        # keep the currency keyword as text if no numeric words found in lookback window
        if not found:
            text += currency + " "

        return text

    def bbc_style_numbers(self, text, number):
        # strip any trailing non-numeric characters
        if not number[-1].isnumeric():
            if number[-1] in ['%', '*', '+', '<', '>', '$', '£', '€']:
                # add word to text
                text += number + " "
                return text
            else:
                number, end_chars = number[:-1], number[-1]
        else:
            end_chars = ""

        # return number to word notation
        formatted_number = num2words(number)

        # capitalise numeric word if at start of sentence
        if text == "" or (len(text) > 2 and text[-2] in TERMINALS):
            formatted_number = formatted_number.capitalize()

        # add word to text
        text += formatted_number + end_chars + " "

        return text

    def insert_number_commas(self, number):
        # strip leading non-numeric characters
        if not number[0].isalnum():
            start_char = number[0]
            number = number[1:]
        else:
            start_char = ""

        # strip any digits after the decimal point in floats
        if "." in number:
            number, end_chars = number.split(".")
            end_chars = "." + end_chars
        else:
            end_chars = ""

        # cycle through number and insert commas every three digits
        start_pos = len(number) - 3
        for i in range(start_pos, 0, -3):
            number = number[:i] + "," + number[i:]

        # reconcatenate leading/trailing chars/digits
        number = start_char + number + end_chars

        return number
