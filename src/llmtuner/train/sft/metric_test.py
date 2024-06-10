import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.packages import (
    is_jieba_available, is_nltk_available, is_rouge_available
)

# transfer
import torch
import torch.nn.functional as F

from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
#from sklearn.metrics.pairwise import cosine_similarity


if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

if is_jieba_available():
    import jieba

if is_nltk_available():
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

if is_rouge_available():
    from rouge_chinese import Rouge


def bleu(hyps, refs):
    """
    bleu
    """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    print(bleu_1)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2

class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": [], "bleu_1": [], "bleu_2": [], "inter_dist1": [], "inter_dist2": [], 
                      "avg_len": [], "avg_label": [], "bleu_11": [], "bleu_22": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            #方式1
            bleu_1 = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method7, weights=[1,0,0,0])
            score_dict["bleu_1"].append(round(bleu_1 * 100, 4))
            bleu_2 = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method7, weights=[0.5,0.5,0,0])
            score_dict["bleu_2"].append(round(bleu_2 * 100, 4))
            
        #方式2
        bleu_11, bleu_22 = bleu(decoded_labels, decoded_preds)
       
        score_dict["bleu_11"].append(round(bleu_11 * 100, 4))
        score_dict["bleu_22"].append(round(bleu_22 * 100, 4))

        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(pred)
        score_dict["inter_dist1"].append(round(inter_dist1 * 100, 4))
        score_dict["inter_dist2"].append(round(inter_dist2 * 100, 4))

    

        return {k: float(np.mean(v)) for k, v in score_dict.items()}