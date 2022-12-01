# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import os
import logging
from langdetect import detect
from simpletransformers.ner import NERModel

PUNCT_LABELS = ['O', '.', ',', ':', ';', "'", '-', '?', '!', '%']
CAPI_LABELS = ['O', 'C', 'U', 'M']
VALID_LABELS = [f"{x}{y}" for y in CAPI_LABELS for x in PUNCT_LABELS]


class RestorePuncts:
    def __init__(self, wrds_per_pred=250, use_cuda=True, model_location='felflare/bert-restore-punctuation'):
        self.model_location = model_location
        self.wrds_per_pred = wrds_per_pred
        self.overlap_wrds = 30
        self.valid_labels = VALID_LABELS
        self.model = NERModel(
            "bert",
            self.model_location,
            labels=self.valid_labels,
            use_cuda=use_cuda,
            args={
                "silent": True,
                "max_seq_length": 512
            }
        )

    def punctuate(self, text: str, lang:str=''):
        """
        Performs punctuation restoration on arbitrarily large text.
        Detects if input is not English, if non-English was detected terminates predictions.
        Overrride by supplying `lang='en'`

        Args:
            - text (str): Text to punctuate, can be few words to as large as you want.
            - lang (str): Explicit language of input text.
        """
        # throw error if text isn't written in english
        if not lang and len(text) > 10:
            lang = detect(text)
        if lang != 'en':
            raise Exception(F"""Non English text detected. Restore Punctuation works only for English.
            If you are certain the input is English, pass argument lang='en' to this function.
            Punctuate received: {text}""")

        # split up large text into bert digestable chunks
        splits = self.split_on_toks(text, self.wrds_per_pred, self.overlap_wrds)

        # predict slices
        full_preds_lst = [self.predict(i['text']) for i in splits]  # full_preds_lst contains tuple of labels and logits (raw predictions)
        preds_lst = [i[0][0] for i in full_preds_lst]  # extract predictions, and discard logits

        # format predictions as a linear sequence of text with per-word predictions tagged
        combined_preds = self.combine_results(text, preds_lst)

        # create full punctuated prediction with correct formatting based upon the predictions
        punct_text = self.punctuate_texts(combined_preds)

        return punct_text

    def predict(self, input_slice):
        """
        Passes the unpunctuated text to the model for punctuation.
        """
        predictions, raw_outputs = self.model.predict([input_slice])

        return predictions, raw_outputs

    @staticmethod
    def split_on_toks(text, length, overlap):
        """
        Splits text into predefined slices of overlapping text with indexes (offsets)
        that tie-back to original text.
        This is done to bypass 512 token limit on transformer models by sequentially
        feeding chunks of < 512 toks.
        Example output:
        [{...}, {"text": "...", 'start_idx': 31354, 'end_idx': 32648}, {...}]
        """
        wrds = text.replace('\n', ' ').split(" ")
        resp = []
        lst_chunk_idx = 0
        i = 0

        while True:
            # words in the chunk and the overlapping portion
            wrds_len = wrds[(length * i):(length * (i + 1))]
            wrds_ovlp = wrds[(length * (i + 1)):((length * (i + 1)) + overlap)]
            wrds_split = wrds_len + wrds_ovlp

            # Break loop if no more words
            if not wrds_split:
                break

            wrds_str = " ".join(wrds_split)
            nxt_chunk_start_idx = len(" ".join(wrds_len))
            lst_char_idx = len(" ".join(wrds_split))

            resp_obj = {
                "text": wrds_str,
                "start_idx": lst_chunk_idx,
                "end_idx": lst_char_idx + lst_chunk_idx,
            }

            resp.append(resp_obj)
            lst_chunk_idx += nxt_chunk_start_idx + 1
            i += 1

        logging.info(f"Sliced transcript into {len(resp)} slices.")
        return resp

    @staticmethod
    def combine_results(full_text: str, text_slices):
        """
        Given a full text and predictions of each slice combines predictions into a single text again.
        Performs validataion wether text was combined correctly
        """
        split_full_text = full_text.replace('\n', ' ').split(" ")
        split_full_text = [i for i in split_full_text if i]  # remove any empty strings
        split_full_text_len = len(split_full_text)
        output_text = []
        index = 0

        # remove final element of prediction list for formatting
        if len(text_slices[-1]) <= 3 and len(text_slices) > 1:
            text_slices = text_slices[:-1]

        # cycle thrugh slices in the full prediction
        for _slice in text_slices:
            slice_wrds = len(_slice)

            # cycle through words in each slice
            for ix, wrd in enumerate(_slice):
                # print(index, "|", str(list(wrd.keys())[0]), "|", split_full_text[index])
                if index == split_full_text_len:
                    break

                # add each (non-overlapping) word and its associated prediction to output text
                if split_full_text[index] == str(list(wrd.keys())[0]) and \
                        ix <= slice_wrds - 3 and text_slices[-1] != _slice:
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)
                elif split_full_text[index] == str(list(wrd.keys())[0]) and text_slices[-1] == _slice:
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)

        # ensure output text content (without predictions) is the same as the full plain text
        assert [i[0] for i in output_text] == split_full_text
        return output_text

    @staticmethod
    def punctuate_texts(full_pred: list):
        """
        Given a list of Predictions from the model, applies the predictions to text,
        thus punctuating it.
        """
        punct_resp = ""

        # cycle through the list containing each word and its predicted label
        for i in full_pred:
            word, label = i

            # implement capitalisation (lowercase/capitalised/uppercase/mixed-case)
            if label[-1] == "U":
                # `xU` => uppercase
                punct_wrd = word.upper()
            elif label[-1] == "C":
                # `xC` => capitalised
                punct_wrd = word.capitalize()
            elif label[-1] == "M":
                # `xM` => mixed-case --- atm just put into uppercase but needs adapting later
                punct_wrd = word.upper()
            else:
                # `xO` => lowercase
                punct_wrd = word

            # if the label indicates punctuation comes after this word, add it
            if label[0] == '-':
                punct_wrd += ' -'
            elif label[0] != "O":
                punct_wrd += label[0]

            punct_resp += punct_wrd + " "

        # remove unnecessary trailing or leading whitespace
        punct_resp = punct_resp.strip()

        # Append trailing period if doesn't exist.
        if punct_resp[-1].isalnum():
            punct_resp += "."
        elif punct_resp[-1] not in ['.', '?', '!']:
            punct_resp = punct_resp[:-1] + "."

        return punct_resp


def run_rpunct(use_cuda=False, input_txt='tests/sample_text.txt', output_txt=None, model_location='felflare/bert-restore-punctuation'):
    # generate instance of rpunct model
    punct_model = RestorePuncts(use_cuda=use_cuda, model_location=model_location)

    # read in txt file file
    print(f"\nReading plaintext from file: {input_txt}")
    try:
        with open(input_txt, 'r') as fp:
            unpunct_text = fp.read()
    except FileNotFoundError:
        input_txt = os.path.join('../', input_txt)
        with open(input_txt, 'r') as fp:
            unpunct_text = fp.read()

    # predict text and print / write out
    punctuated = punct_model.punctuate(unpunct_text)

    if output_txt is None:
        # print output to command line
        print("\nPrinting punctuated text", end='\n\n')
        print(punctuated)
    else:
        # check if output directory exists
        output_path, output_file = os.path.split(output_txt)
        output_path_exists = os.path.isdir(output_path)

        # print punctuated text to output file
        if output_path_exists or output_path == '':
            print(f"Writing punctuated text to file: {output_txt}")
            with open(output_txt, 'w') as fp:
                fp.write(punctuated)
        else:
            raise FileNotFoundError("Directory specified to ouptut text file to does not exist.")


if __name__ == "__main__":
    cuda = False
    input = 'tests/sample_text.txt'
    output = 'output.txt'
    run_rpunct(use_cuda=cuda, input_txt=input, output_txt=output)
