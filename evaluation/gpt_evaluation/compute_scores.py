import argparse
import concurrent.futures
import json
import os
import time
import sys
from openai import OpenAI
import tenacity
import re

from datasets import load_dataset

from evaluator import Evaluator
from tqdm.auto import tqdm
from utils import get_azure_response, parse_output
from utils import process_data_list


def run(content, n):
    retry_numbers = 0
    while True:
        retry_numbers += 1
        response = get_azure_response(
            url      = url,
            apikey   = apikey,
            content  = content,
            n        = n,
        )

        all_scores = [parse_output(r) for r in response]

        count = 0
        scores = 0
        for score in all_scores:
            if score > 0 and score <= 5:
                count += 1
                scores += score

        if count >= 2/3 * n or (retry_numbers >= 5 and count > 0):
            break

    return scores / count

def get_dataset(path):
    dataset = load_dataset(
        'json',
        data_files = path,
        split      = 'train'
    )

    inputs = dataset['input']
    predicts = dataset['predict']

    return inputs, predicts, dataset

parser = argparse.ArgumentParser(description="evaluate by chatgpt")

parser.add_argument('--type', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--output_path', type=str, default='scores.json')
parser.add_argument('--url', type=str)
parser.add_argument('--apikey', type=str)

args = parser.parse_args()

# python gpt_evaluation/compute_scores.py --type fluency --data_path datas1/chatgpt_gen_woper.json > ./logs1/chatgpt_gen_fluency.log 2>&1
# python gpt_evaluation/compute_scores.py --type interestingness --data_path datas1/genllama.json > ./logs/genllama_interestingness.log 2>&1

if __name__ == '__main__':
    apikey = args.apikey
    url = args.url

    inputs, predicts, dataset = get_dataset(args.data_path)

    evaluator = Evaluator(type=args.type)
    queries = evaluator.make_queries(
        inputs=inputs,
        predicts=predicts
    )
    print(queries[3])
    # time.sleep(100)

    # response = get_azure_response(content="深度学习")
    client = OpenAI(
        base_url="https://neudm.zeabur.app/v1",
        api_key="sk-PpBsrIQR2GU2BVWX5aB993515b5644E9A82d17052695B6Bf"
    )
    # client = OpenAI(
    #     api_key="a6000",
    #     base_url="http://219.216.65.153:9099/v1/",
    # )

    scores_gen = []
    for query in queries:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        )
        response = completion.choices[0].message.content
        scores_gen.append(response)
    print(scores_gen)

    count, total_sum = process_data_list(scores_gen)

    # 打印结果
    print("符合条件的项数:", count)
    print("符合条件的项中的数字总和:", total_sum)
    print("score",total_sum/count)

    # fp = open(os.path.join('gpt_evaluation/result', args.output_path), 'a', encoding='utf-8')
    # total_scores = 0
    # total_count = 0
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = [
    #         executor.submit(
    #             run,
    #             content = query,
    #             n       = 2
    #         ) for query in queries
    #     ]
    #
    #     with tqdm(total=len(futures)) as pbar:
    #         for future, data in zip(concurrent.futures.as_completed(futures), dataset):
    #             try:
    #                 data['scores'] = future.result()
    #                 total_scores += data['scores']
    #                 total_count += 1
    #                 fp.write(
    #                     json.dumps(data, ensure_ascii=False) + '\n'
    #                 )
    #                 pbar.update(1)
    #                 pbar.set_postfix({f'{args.type.upper()} Average Score': total_scores/total_count})
    #             except Exception as e:
    #                 print(e)
    #                 pbar.update(1)
    #
    # print(f'{args.data_path}')
    # # print(f'{args.type} score: {total_scores / total_count}')
    # print(total_scores, total_count)
    #
    # with open('./result/result.json', 'a') as w:
    #     w.write(
    #         json.dumps({
    #             'data': args.data_path,
    #             'type': args.type,
    #             'score': total_scores / total_count
    #         }, ensure_ascii=False) + '\n'
    #     )




