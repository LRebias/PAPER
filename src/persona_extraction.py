from openai import OpenAI
import json


def generate_completions_and_write(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r') as file:
        data = json.load(file)

    client = OpenAI(
        base_url="xxxxx",
        api_key="xxxxx"
    )

    # 处理每个样例
    completions = []
    for example in data:
        # 合并instruction和input部分
        content = example['instruction'] + example['input']

        # 生成completion
        completion = client.chat.completions.create(
            model="XXX",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]
        )

        # 添加completion到列表
        completions.append(completion.choices[0].message.content)

    # 写入生成的每一个completion到JSON文件中
    with open(output_file, 'w') as output:
        json.dump(completions, output)

def persona_extract(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r') as file:
        data = json.load(file)

    client = OpenAI(
        api_key="xxxxx",
        base_url="xxxxx",
    )

    # 处理每个样例
    persona_generate = []
    for example in data:
        # 合并instruction和input部分
        content = "This is a multi-party conversation.Give the most accurate one-sentence short description of speaker's personality in the context of speaker's utterances in the history of the dialogue." \
                  + example['input']

        # 生成completion
        completion = client.chat.completions.create(
            model="XXX",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]
        )

        # 添加completion到列表
        persona_generate.append(completion.choices[0].message.content)

    # 写入生成的每一个completion到JSON文件中
    with open(output_file, 'w') as output:
        json.dump(persona_generate, output, indent=4)
persona_extract("datas/test_woper.json", "datas/persona_glm3.json")

