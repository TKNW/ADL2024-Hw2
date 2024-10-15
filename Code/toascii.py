import jsonlines
import json
data_list = []

with jsonlines.open('./prediction.jsonl') as reader:
    for obj in reader:
        data_list.append(obj)
with open("./pred_asc.jsonl", mode='w',encoding="utf-8") as writer:
    for entry in data_list:
        # 使用 ensure_ascii=True 將非ASCII字符轉成 \uXXXX 格式
        json_str = json.dumps(entry, ensure_ascii=True)
        # 直接寫入 json_str，並加上換行符號以符合 jsonlines 格式
        writer.write(json_str + '\n')