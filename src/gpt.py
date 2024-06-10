from openai import OpenAI
import json


def generate_completions_and_write(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r') as file:
        data = json.load(file)

    # 初始化OpenAI客户端
    # client = OpenAI(
    #     base_url="https://oneapi.xty.app/v1",
    #     api_key="sk-LFYwP7Z5aia9zUwe8745B5063b28474aBc52910b3aAd8eCb"
    # )
    client = OpenAI(
        base_url="https://neudm.zeabur.app/v1",
        api_key="sk-PpBsrIQR2GU2BVWX5aB993515b5644E9A82d17052695B6Bf"
    )

    # 处理每个样例
    completions = []
    for example in data:
        # 合并instruction和input部分
        content = example['instruction'] + example['input']

        # 生成completion
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
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
        api_key="a6000",
        base_url="http://219.216.64.109:9099/v1/",
    )

    # 处理每个样例
    persona_generate = []
    for example in data:
        # 合并instruction和input部分
        content = "This is a multi-party conversation.Give the most accurate one-sentence short description of speaker's personality in the context of speaker's utterances in the history of the dialogue." \
                  + example['input']

        # 生成completion
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
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
# 使用函数生成completion并写入文件
# generate_completions_and_write('a_test100.json', 'LLama_completions.json')
# persona_extract("datas/test_woper_example.json", "datas/persona_generate1.json")
# generate_completions_and_write('datas/preception_test.json','datas/preception_gpt_gen.json')
persona_extract("datas/test_woper.json", "datas/persona_glm3.json")

# print(completion.choices[0].message)
# # print("1",completion['choices'][0]['message']['content'])
# print(completion.choices[0].message.content)

# response = openai.ChatCompletion.create(
#   engine="xpy_gpt35",
#   messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"你好"},{"role":"assistant","content":"你好！有什么我可以帮助你找到的信息吗？"}],
#   temperature=0.7,
#   max_tokens=800,
#   top_p=0.95,
#   frequency_penalty=0,
#   presence_penalty=0,
#   stop=None)
# print(response)
