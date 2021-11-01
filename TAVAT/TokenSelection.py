from typing import List, Dict

import nltk
from tqdm import tqdm
import logging
from transformers import pipeline
import random
tqdm.pandas()

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

nltk.download("wordnet")
logger = logging.getLogger(__name__)


class PivotTokenSelector:
    @staticmethod
    def fill_mask(text, label):
        LEAVE_OUT_WORDS = {'understand', 'event', 'important', 'know', 'statement', 'true'}

        aug_sents_dicts = []
        text = nltk.word_tokenize(text)
        result = nltk.pos_tag(text)
        for idx, (word, tag) in enumerate(result):
            if word.lower() in LEAVE_OUT_WORDS:
                continue
            tmp_result = [list(ele) for ele in result]
            masked_word = tmp_result[idx][0]
            tmp_result[idx][0] = "[MASK]"
            orig_sentence = text
            if PivotTokenSelector.valid_pos_tag(tag):
                new_aug_dict = {
                    'original_sentence': text,
                    'masked_word': masked_word,
                    'masked_position': idx,
                    'label': label,
                    'masked_sentence': ' '.join(word[0] for word in tmp_result),
                }
                aug_sents_dicts.append(new_aug_dict)

        return aug_sents_dicts

    @staticmethod
    def aug_selection_strategy(aug_sents_dicts: List[Dict], n_samples: int):
        """
            Selection strategy for selecting the augmented sentences.
            Current strategy is selecting a max of 15 sentences at random.
        :param aug_sents_dicts:
        :param n_samples:
        :return:
        """
        return random.sample(aug_sents_dicts, min(n_samples, len(aug_sents_dicts)))

    # Return true if the pos tag is either a Noun or an Adjective.
    @staticmethod
    def valid_pos_tag(tag):
        NOUN_CODES = {'NN', 'NNS', 'NNPS', 'NNP'}
        ADJECTIVE_CODES = {"JJ", "JJR", "JJS"}
        if (tag in NOUN_CODES) or (tag in ADJECTIVE_CODES):
            return True
        return False