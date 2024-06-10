import json

def update_personality(file1_path, file2_path, output_file_path):
    # 读取第一个json文件
    with open(file1_path, 'r') as file:
        data1 = json.load(file)

    # 读取第二个json文件
    with open(file2_path, 'r') as file:
        personas = json.load(file)

    # 确保persona列表的长度与data1中的条目数相同
    assert len(data1) == len(personas), "The number of elements in both files should match"

    # 更新data1中的input字段
    for i, entry in enumerate(data1):
        parts = entry['input'].split("\t")
        # personality是第二部分，将新的personality添加到现有的后面
        parts[1] += f", {personas[i]}"
        # 重新组合input字符串
        entry['input'] = "\t".join(parts)

    # 将更新后的数据写入新的json文件
    with open(output_file_path, 'w') as file:
        json.dump(data1, file, indent=4)

    print(f"The data has been updated and written to '{output_file_path}'.")

def replace_personality(file1_path, file2_path, output_file_path):
    # 读取第一个json文件
    with open(file1_path, 'r') as file:
        data1 = json.load(file)

    # 读取第二个json文件
    with open(file2_path, 'r') as file:
        personas = json.load(file)

    # 确保persona列表的长度与data1中的条目数相同
    assert len(data1) == len(personas), "The number of elements in both files should match"

    # 替换data1中的input字段中的personality部分
    for i, entry in enumerate(data1):
        parts = entry['input'].split("\t")
        # 固定personality为第二部分，直接替换
        parts[1] = f"personality:{personas[i]}"
        # 重新组合input字符串
        entry['input'] = "\t".join(parts)

    # 将更新后的数据写入新的json文件
    with open(output_file_path, 'w') as file:
        json.dump(data1, file, indent=4)

    print(f"The data has been replaced and written to '{output_file_path}'.")


