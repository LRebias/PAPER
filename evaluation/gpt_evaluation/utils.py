from openai import OpenAI
import tenacity
import re

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=200),
    stop=tenacity.stop_after_attempt(3),
    reraise=True)
def get_azure_response(
    content: str
):
    # openai.api_type    = "azure"
    # openai.api_base    = url
    # openai.api_version = "2023-03-15-preview"
    # openai.api_key     = apikey
    client = OpenAI(
        api_key="sk-QQbKT7M6kCMIyk8z2fC920057e7f4834Ab90A9B6F57b3030",
        base_url="https://api.bear0225.xyz/v1/",
    )

    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages = [
            {
                "role"   : "system",
                "content": "You are a helpful assistant."
            },
            {
                "role"   : "user",
                "content": content
            }
        ]
    )

    # all_responses = [response.choices[i].message.content
    #     for i in range(len(response.choices))]
    all_responses = response.choices[0].message.content

    return all_responses

def parse_output(output):
    matched = re.search("^ ?([\d\.]+)", output)
    if (matched):
        try:
            score = float(matched.group(1))
        except:
            score = -1
    else:
        score = -1
    return score

def extract_first_matching_number(item):
    # 新的正则表达式模式，匹配1到5的数字
    # 匹配只有数字的情况
    digit_match = re.match(r'^([\d\.]+)$', item)
    if digit_match:
        return float(digit_match.group(1))

    # 匹配任意字符串(1-5):5这种格式
    generic_match = re.match(r'.*\(1-5\):\s*(\d+)$', item)
    if generic_match:
        return float(generic_match.group(1))

    # 如果无法匹配，返回 None 或其他适当的默认值
    return None

def process_data_list(data_list):
    # 初始化变量
    count_matching_items = 0
    sum_of_matching_numbers = 0

    # 遍历列表中的每一项
    for item in data_list:
        # 提取每项中的第一个符合条件的数字
        matched_number = extract_first_matching_number(item)

        # 如果找到匹配的数字
        if matched_number is not None:
            # 计算符合条件的项数
            count_matching_items += 1

            # 计算符合条件的项中的数字总和
            sum_of_matching_numbers += matched_number

    # 返回结果
    return count_matching_items, sum_of_matching_numbers

# 给定的列表
data_list = ['2', '3', 'the score is 4', '3', 'score 3 is ok', '5']

# 调用函数并获取结果
count, total_sum = process_data_list(data_list)

# 打印结果
print("符合条件的项数:", count)
print("符合条件的项中的数字总和:", total_sum)
