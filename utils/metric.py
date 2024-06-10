import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# transfer
import torch
import torch.nn.functional as F

from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
# from sklearn.metrics.pairwise import cosine_similarity
import json
import jieba
from rouge_chinese import Rouge
if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
import json
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity


# transfer
def accuracy(logits, targets, padding_idx=None):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    _, preds = logits.max(dim=2)
    trues = (preds == targets).float()
    if padding_idx is not None:
        weights = targets.ne(padding_idx).float()
        acc = (weights * trues).sum(dim=1) / weights.sum(dim=1)
    else:
        acc = trues.mean(dim=1)
    acc = acc.mean()
    return acc


def attn_accuracy(logits, targets):
    """
    logits: (batch_size, vocab_size)
    targets: (batch_size)
    """
    _, preds = logits.squeeze(1).max(dim=-1)
    trues = (preds == targets).float()
    acc = trues.mean()
    return acc


def perplexity(logits, targets, weight=None, padding_idx=None):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    batch_size = logits.size(0)
    if weight is None and padding_idx is not None:
        weight = torch.ones(logits.size(-1))
        weight[padding_idx] = 0
    nll = F.nll_loss(input=logits.view(-1, logits.size(-1)),
                     target=targets.contiguous().view(-1),
                     weight=weight,
                     reduction='none')
    nll = nll.view(batch_size, -1).sum(dim=1)
    if padding_idx is not None:
        word_cnt = targets.ne(padding_idx).float().sum()
        nll = nll / word_cnt
    ppl = nll.exp()
    return ppl


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


def distinct(seqs):
    """
    distinct
    """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        # unigrams = Counter(seq)
        # bigrams = Counter(zip(seq, seq[1:]))
        # intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        # intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))
        unigrams = Counter(seq.split())  # Distinct-1
        bigrams = Counter(zip(seq.split(), seq.split()[1:]))  # Distinct-2

        intra_dist1.append((len(unigrams) + 1e-12) / (len(seq.split()) + 1e-5))
        intra_dist2.append((len(bigrams) + 1e-12) / (max(0, len(seq.split()) - 1) + 1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all) + 1e-12) / (sum(unigrams_all.values()) + 1e-5)
    inter_dist2 = (len(bigrams_all) + 1e-12) / (sum(bigrams_all.values()) + 1e-5)
    # intra_dist1 = np.average(intra_dist1)
    # intra_dist2 = np.average(intra_dist2)
    intra_dist1 = sum(intra_dist1) / batch_size
    intra_dist2 = sum(intra_dist2) / batch_size
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def cosine(X, Y):
    """
    cosine
    """
    sim = np.sum(X * Y, axis=1) / \
          (np.sqrt((np.sum(X * X, axis=1) * np.sum(Y * Y, axis=1))) + 1e-10)
    return sim


def Knowledge_R_P_F1(hyps, cues):
    sw_counter = set()
    for line in open('./untils/stopwords.txt', 'r', encoding='utf-8'):
        sw_counter.add(line.strip())
    R_list = []
    P_list = []
    F1_list = []
    for hyp, cue in zip(hyps, cues):

        cue_counter = set()
        for line in cue:
            cue_counter.update(line.strip().split(' '))
        hyp_counter = set(hyp)

        cue_counter -= sw_counter
        if len(cue_counter) == 0:
            continue
        hyp_counter -= sw_counter

        r = (cue_counter & hyp_counter).__len__() / (cue_counter.__len__() + 0.00001)
        R_list.append(r)
        p = (cue_counter & hyp_counter).__len__() / (hyp_counter.__len__() + 0.00001)
        P_list.append(p)
        if r == 0 or p == 0:
            f1 = 0
        else:
            f1 = 2 * r * p / (r + p)
        F1_list.append(f1)
    return np.average(R_list), np.average(P_list), np.average(F1_list)

def calculate_embedding_average(sentence,tokenizer1,bert_model1):
    tokens = tokenizer1(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model1(**tokens)
    last_hidden_states = outputs.last_hidden_state
    sentence_vector = last_hidden_states.mean(dim=1).squeeze().numpy()
    return sentence_vector


def calculate_embedding_greedy_match(reference, generated,tokenizer1,bert_model1):
    ref_tokens = tokenizer1(reference, return_tensors='pt', truncation=True, padding=True)
    gen_tokens = tokenizer1(generated, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        ref_outputs = bert_model1(**ref_tokens)
        gen_outputs = bert_model1(**gen_tokens)

    ref_last_hidden_states = ref_outputs.last_hidden_state
    gen_last_hidden_states = gen_outputs.last_hidden_state
    similarity_sum = 0.0

    for ref_token in ref_last_hidden_states[0]:
        max_similarity = max([cosine_similarity([ref_token.numpy()], [gen_token.numpy()])[0, 0] for gen_token in
                              gen_last_hidden_states[0]])
        similarity_sum += max_similarity

    return similarity_sum / len(ref_last_hidden_states[0])


def calculate_embedding_extremes(sentence,tokenizer1,bert_model1):
    tokens = tokenizer1(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model1(**tokens)
    last_hidden_states = outputs.last_hidden_state

    s_max = np.max(last_hidden_states.numpy(), axis=1).squeeze()
    s_min = np.min(last_hidden_states.numpy(), axis=1).squeeze()
    s_plus = np.abs(s_min) <= s_max
    sentence_vector = s_max * s_plus + s_min * ~s_plus
    # sentence_vector = s_max * s_plus + s_min * np.logical_not(s_plus)

    return sentence_vector


def calculate_cosine_similarity(vector_a, vector_b):
    similarity = cosine_similarity([vector_a], [vector_b])[0, 0]
    return similarity

with open('untils/preception_gpt_gen.json', 'r') as file:
    data1 = json.load(file)
decoded_preds = data1
print(decoded_preds)

with open('data/preception_test.json', 'r') as file:
    data2 = json.load(file)
decoded_labels =[]
for example in data2:
    decoded_labels.append(example['output'])

score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": [], "bleu_1": [], "bleu_2": [], "inter_dist1": [],
              "inter_dist2": [],
              "avg_len": [], "avg_label": [], "bleu_11": [], "bleu_22": [], "R": [], "P": [], "F1": [],"emb_e": [],"emb_a": [],"emb_g": []}

# 测试
# print(decoded_preds)
for pred, label in zip(decoded_preds[:3], decoded_labels[:3]):
    print("1:", pred)
    print("2:", label)

for pred, label in zip(decoded_preds, decoded_labels):
    hypothesis = list(jieba.cut(pred))
    reference = list(jieba.cut(label))

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    for k, v in result.items():
        score_dict[k].append(round(v["f"] * 100, 4))

    bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
    score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    # add

    bleu_1 = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method7,
                           weights=[1, 0, 0, 0])
    score_dict["bleu_1"].append(round(bleu_1 * 100, 4))
    bleu_2 = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method7,
                           weights=[0.5, 0.5, 0, 0])
    score_dict["bleu_2"].append(round(bleu_2 * 100, 4))
    avg_len = len(pred.split())
    score_dict["avg_len"].append(avg_len)
    avg_label = len(label.split())
    score_dict["avg_label"].append(avg_label)

print(score_dict["bleu_1"])

preds_split = [sentence.split() for sentence in decoded_preds]
# intra_dist11, intra_dist22, inter_dist11, inter_dist22 = distinct(preds_split)
# print("dist",inter_dist11,inter_dist22)


bleu_11, bleu_22 = bleu(decoded_preds, decoded_labels)
score_dict["bleu_11"].append(round(bleu_11 * 100, 4))
score_dict["bleu_22"].append(round(bleu_22 * 100, 4))

intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(decoded_preds)
score_dict["inter_dist1"].append(round(inter_dist1 * 100, 4))
score_dict["inter_dist2"].append(round(inter_dist2 * 100, 4))
print("intra_dist", intra_dist1, intra_dist2)

with open('untils/a_test-t.json', 'r') as file:
    data = json.load(file)

# 提取input中的personality内容
personas = []
for item in data:
    input_parts = item['input'].split('\t')
    for part in input_parts:
        if part.startswith('personality:'):
            personality = part[len('personality:'):]
            personas.append(personality)

R, P, F1 = Knowledge_R_P_F1(decoded_preds, personas)
score_dict["R"].append(round(R * 100, 4))
score_dict["P"].append(round(P * 100, 4))
score_dict["F1"].append(round(F1 * 100, 4))

persona_split = [sentence.split() for sentence in personas]
RR, PP, FF1 = Knowledge_R_P_F1(preds_split, persona_split)
print("persona", RR, PP, FF1)

model_name = '/datas/huggingface/bert-base-uncased'
tokenizer1 = BertTokenizer.from_pretrained(model_name)
bert_model1 = BertModel.from_pretrained(model_name)
for ref_sent, gen_sent in zip(decoded_labels, decoded_preds):
    # Emb A
    emb_a_reference = calculate_embedding_average(ref_sent,tokenizer1,bert_model1)
    emb_a_generated = calculate_embedding_average(gen_sent,tokenizer1,bert_model1)
    score_dict["emb_a"].append(calculate_cosine_similarity(emb_a_reference, emb_a_generated))

    # Emb G
    score_dict["emb_g"].append(calculate_embedding_greedy_match(ref_sent, gen_sent,tokenizer1,bert_model1))

    # Emb E (max)
    emb_e_reference = calculate_embedding_extremes(ref_sent,tokenizer1,bert_model1)
    emb_e_generated = calculate_embedding_extremes(gen_sent,tokenizer1,bert_model1)
    score_dict["emb_e"].append(calculate_cosine_similarity(emb_e_reference, emb_e_generated))

print({k: float(np.mean(v)) for k, v in score_dict.items()})