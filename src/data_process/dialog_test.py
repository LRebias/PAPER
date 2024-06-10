import json

def convert_data(raw_data):
    converted_data = []
    for data in raw_data:

        data_parts = data.split('\t')
        if len(data_parts) >= 4:
            context = data_parts[0]
            response = data_parts[1]
            persona = data_parts[2]
            responder_persona = data_parts[3]
            # 继续处理剩余的代码
            # src, tgt, persona, responder_persona = data.split('\t')[:4]

            # src = src.split("[u_sep]")
            # persona = persona.split("[p_sep]")
            # responder_persona = responder_persona.split("[resp_sep]")

        else:
            print("error")

        input_text = "dialog history:" + context  + '\t' + "personality:" + persona  + '\t' + "responder_personality:" + responder_persona
        # input_text = "\n".join(src + persona + responder_persona)
        output_text = response.strip()

        conversation = {
            "instruction": "This is a context of a multi-party ",
            "input": input_text,
            "output": output_text
        }

        converted_data.append(conversation)

    return converted_data
def load_raw_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = file.readlines()
    return raw_data

file_path = 'data/demo.test'
raw_data = load_raw_data(file_path)
converted_data = convert_data(raw_data)

# raw_data = [
#     "(At the party, Sydney walks and takes the coin out of her purse.)[u_sep]MARSHALL: (voice over) Just make sure you drop it near a window.[u_sep](She does so. She looks around, walks, mingles in the crowd. Sydney feels someone's eyes on her, looks up, and sees Ana posing as a waitress. They make eye contact. Ana winks at Sydney.)[u_sep]SYDNEY: Ana just crashed the party.[u_sep](Dixon's inside a van with camera screens up everywhere.)	Careful, Syd.	[p_sep]MARSHALL OddFriendship PhotographicMemory MARSHALL: (voice over) Just make sure you drop it near a window.[p_sep]SYDNEY: I'm in. SYDNEY: Ana just crashed the party.	AchillesInHisTent[resp_sep]BewareTheNiceOnes[resp_sep]CoolOldGuy",
#     "(Doorbell rings.)[u_sep]KIDS: Trick or treat![u_sep]SYDNEY: Hey, guys! There's more candy in there. Come on in![u_sep]DIANE: Good to see you.[u_sep]SYDNEY: Good to see you.[u_sep]DIXON: Syd, can I talk to you for a second?[u_sep](They go in anothe room, close the door.)	What's up?	SYDNEY: Good to see you.[p_sep]DIXON AchillesInHisTent BewareTheNiceOnes DIXON: Syd, can I talk to you for a second?[p_sep]KIDS: Trick or treat!	Omniglot[resp_sep]ScaryBlackMan[resp_sep]TheSpymaster"
#     ]

# converted_data = convert_data(raw_data)

# 写入JSON文件
output_file_path = 'data/dialog_test.json'
with open(output_file_path, 'w') as file:
    json.dump(converted_data, file, indent=4)

print("转换后的数据已写入JSON文件:", output_file_path)
# print(json.dumps(converted_data, indent=4))