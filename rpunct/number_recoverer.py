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


STRING_NUMBERS = ['million', 'billion', 'trillion']
currencies = {
    'pound': '£',
    'euro': '€',
    'dollar': '$'
}


class NumberRecoverer:
    """
    Parent class for number recovery. Uses `number_parser` (https://pypi.org/project/number-parser/)
    to convert numbers written in the natural language to their equivalent numeric forms.
    """

    def __init__(self, correct_currencies=True, correct_bbc_style_numbers=True, comma_separators=True):
        self.correct_currencies = correct_currencies
        self.correct_bbc_style_numbers = correct_bbc_style_numbers
        self.comma_separators = comma_separators

    def process(self, text):
        """
        Pipeline for recovering formatting of numbers within a piece of text.
        """
        # Convert numerical strings to digits in text using `number_parser` package
        parsed_text = self.number_parser(text)

        # Convert percentages to use the symbol notation
        parsed_text = parsed_text.replace(" percent", "%")

        # If we are correcting currencies, we don't want these words to be hidden mid-hyphenation
        if self.correct_currencies:
            for word in currencies.keys():
                parsed_text = parsed_text.replace(f"-{word}", f" {word}")

        # Restore decimal points
        parsed_list = parsed_text.split(" ")
        parsed_list = self.replace_decimal_points(parsed_list)

        # Correct currencies, BBC styling of numbers, and insert currency separators into numbers >= 10,000
        output_text = ""
        for word in parsed_list:
            stripped_word = re.sub(r"[^0-9a-zA-Z]", "", word).lower()

            # Restore currency words to their symbols
            if self.correct_currencies and self.is_currency(stripped_word):
                    output_text = self.insert_currency_symbols(output_text, word)

            # BBC Style Guide asserts that single digit numbers should be written as words, so revert those numbers
            elif self.correct_bbc_style_numbers and self.is_stylable_number(word):
                    output_text = self.bbc_style_numbers(output_text, word)

            # Format numbers with many digits to include comma separators
            elif self.comma_separators and stripped_word.isnumeric() and int(stripped_word) >= 10000:
                output_text += self.insert_comma_seperators(word) + " "

            else:
                output_text += word + " "

        # Remove any unwanted whitespace
        output_text = output_text.strip()
        output_text = output_text.replace(" - ", "-")
        output_text = output_text.replace("- ", "-")

        return output_text

    def number_parser(self, text):
        """
        Converts numbers in text to digits instead of words.
        Optionally very large (>= million) numbers can stay as words.
        """
        # BBC Style Guide asserts that single digit numbers should be written as words, so don't convert those numbers
        # also ensure large number definitions remain as words in text
        if self.bbc_style_numbers:
            # Swap digits that we don't want to be parsed with control characters (from STRING_NUMBERS lookup table)
            control_chars = list(enumerate(STRING_NUMBERS))
            for index, number in control_chars:
                text = text.replace(number, f"\\{index}")

            # Convert all other numbers to digits
            parsed = number_parser(text)

            # Return control characters to words
            control_chars.reverse()
            for index, number in control_chars:
                parsed = parsed.replace(f"\\{index}", number)
        else:
            parsed = number_parser(text)

        # `number_parser` adds spaces around numbers, interrupting the formatting of any trailing punctuation, so re-concatenate
        for punct in string.punctuation:
            parsed = parsed.replace(f" {punct}", f"{punct}")

        return parsed

    def is_currency(self, word):
        """Checks if a word is a currency term."""
        return (word in currencies.keys()) or (word[-1] == 's' and word[:-1] in currencies.keys())

    def is_stylable_number(self, number):
        """Checks if a number is single digit and should be converted to a word according to the BBC Style Guide."""
        # (Includes failsafe if number is immediately followed by a punctuation character)
        return (number.isnumeric() and int(number) < 10) or (not number[-1].isnumeric() and number[:-1].isnumeric() and int(number[:-1]) < 10)

    @staticmethod
    def replace_decimal_points(text_list):
        """
        Correctly format numbers with decimal places (e.g. "1 point 5" -> "1.5").
        """
        corrected_list = []
        i = 0

        while i < len(text_list):
            # Cycle through words in the text until the word "point" appears (can't be the 1st or last word)
            word = text_list[i]

            if re.sub(r"[^0-9a-zA-Z]", "", word) == "point" and i > 0 and i < len(text_list) - 1:
                # When a case for decimal point formatting is identified, combine stripped full no. and decimal digits together around a `.` char
                pre_word = text_list[i - 1]
                pre_word_stripped = re.sub(",.?!%", "", pre_word)
                post_word = text_list[i + 1]
                post_word_stripped = re.sub(r"[^0-9a-zA-Z]", "", post_word)

                # Ensure both words around the deminal point are numerical
                # N.B. concatenate the original (not stripped) post word s.t. any trailing punctuation is preserved
                if pre_word_stripped.isnumeric() and post_word_stripped.isnumeric():
                    full_number = pre_word_stripped + '.' + post_word
                    corrected_list = corrected_list[:-1]
                    corrected_list.append(full_number)
                    i += 2

                # All other words are simply added back into the text
                else:
                    corrected_list.append(word)
                    i += 1
            else:
                corrected_list.append(word)
                i += 1

        return corrected_list

    def insert_currency_symbols(self, text, currency):
        """
        Converts currency terms in text to symbols before their respective numerical values.
        """
        # Get plaintext version of currency keyword
        stripped_currency = re.sub(r"[^0-9a-zA-Z]", "", currency).lower()
        if stripped_currency[-1] == 's':
            stripped_currency = stripped_currency[:-1]

        found = False
        lookback = 1

        # Scan through lookback window to find the number to which the currency symbol punctuates
        text_list = text.split(" ")
        while lookback < 5:
            prev_word = text_list[-lookback]
            prev_word_stripped = re.sub(r"[^0-9a-zA-Z]", "", prev_word)

            # When a numeric word is found, reconstruct the output text around this (i.e. previous_text + currency_symbol + number + trailing_text)
            if prev_word_stripped.isnumeric():
                new_output_text = text_list[:-lookback]  # previous text before currency symbol
                new_output_text.append(currencies.get(stripped_currency) + prev_word)  # currency symbol and number
                new_output_text.extend(text_list[-lookback + 1:])  # trailing text after currency symbol/number
                text = " ".join(new_output_text)

                # Add back in any punctuation trailing the original currency keyword
                if not currency[-1].isalnum():
                    text = text[:-1] + currency[-1] + " "

                found = True
                break
            else:
                lookback += 1

        # Keep the currency keyword as text if no numeric words found in lookback window
        if not found:
            text += currency + " "

        return text

    def bbc_style_numbers(self, text, number):
        """
        Converts small numbers back from digits to words (according to BBC Style Guide rules).
        """
        if not number[-1].isnumeric():
            # Don't convert number if it is involved with some mathematical/currency expression
            if number[-1] in ['%', '*', '+', '<', '>', '$', '£', '€']:
                text += number + " "
                return text
            # But separate off any trailing punctuation other than this and continue
            else:
                number, end_chars = number[:-1], number[-1]
        else:
            end_chars = ""

        # Return number to word notation
        formatted_number = num2words(number)

        # Capitalise numeric word if at start of sentence
        if text == "" or (len(text) > 2 and text[-2] in TERMINALS):
            formatted_number = formatted_number.capitalize()

        # Add word to text
        text += formatted_number + end_chars + " "

        return text

    def insert_comma_seperators(self, number):
        """
        Inserts comma separators into numbers with many digits to break up 1000s (e.g. '100000' -> '100,000').
        """
        # Strip leading non-numeric characters and trailing digits after the decimal point in floats
        if not number[0].isalnum():
            start_char = number[0]
            number = number[1:]
        else:
            start_char = ""

        if "." in number:
            number, end_chars = number.split(".")
            end_chars = "." + end_chars
        else:
            end_chars = ""

        # Cycle through number in reverse order and insert comma separators every three digits
        for i in range(len(number) - 3, 0, -3):
            number = number[:i] + "," + number[i:]

        # Reconcatenate leading/trailing chars/digits
        number = start_char + number + end_chars

        return number
