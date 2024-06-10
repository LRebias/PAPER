import json

def clean_data(data):
    cleaned_data = []
    for item in data:
        # Check and filter conditions for 'output'
        output_words = len(item['output'].split())
        if 3 <= output_words <= 50:
            parts = item['input'].split('\t')
            if len(parts) == 3:
                context_text, persona_text, responder_persona_text = parts

                # Check and filter conditions for 'context' and 'persona'
                if 3 <= context_text.count(';') <= 10 and persona_text.count(':') <= 4:
                    cleaned_item = {
                        "instruction": item["instruction"],
                        "input": f" {context_text}\t{persona_text}\t{responder_persona_text}",
                        # "input": f"{context_text}",
                        "output": item["output"]
                    }
                    cleaned_data.append(cleaned_item)
    return cleaned_data

# 读取原始JSON文件
with open('data\dialog_test.json', 'r') as file:
    data = json.load(file)

# 修改数据
for item in data:
    # item["instruction"] = " This is a multi-party conversation. Please think of a more understandable and rich explanation of persona based on the given personality. Then generate a short response based on the dialog history, the personality of the speaker, and the responder's personality. This response is required to be consistent with the speaker's personality information."
    item["instruction"] = " This is a multi-party conversation. Please generate a short response based on the dialog history, the personality of the speaker, and the responder's personality. This response is required to be consistent with the speaker's personality information."
    # item["instruction"] = " This is a multi-party conversation. Please generate a short response based on the dialog history."
    input_text = item["input"]
    input_text = input_text.replace("[p_sep]", " ; ").replace("[u_sep]", " ; ").replace("[resp_sep]", " ; ")
    item["input"] = input_text

cleaned_data = clean_data(data)

# 写入新的JSON文件
with open('data1/a_test-t.json', 'w') as file:
    json.dump(cleaned_data, file, indent=4)

print("清理后的数据个数:", len(cleaned_data))