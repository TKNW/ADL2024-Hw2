import argparse
import json, random
from datetime import datetime
import langid
import emoji
from tqdm import tqdm
import re


parser = argparse.ArgumentParser(description="Preprocess data.")
parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file.",
    )
parser.add_argument(
        "--output_train",
        required=True,
        type=str,
        help="Output train file.",
    )
parser.add_argument(
        "--output_valid",
        type=str,
        required=True,
        help="Output validation file.",
    )
parser.add_argument(
        "--output_remove",
        type=str,
        help="Output remove column file.",
    )
parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="train data size",
)
parser.add_argument(
        "--prompt",
        action="store_true",
        help="Determine use prompt or not",
)
args = parser.parse_args()
with open(args.input, 'r', encoding='utf-8') as f:
    rawdata = json.load(f)

data = []
final_data = []
remove_data = []
count = 0
print("Remove non-zh data...")
for article in tqdm(rawdata):
    language, score = langid.classify(article['maintext'])
    if language != "zh":
        remove_data.append(article)
        count += 1
    else:
        data.append(article)

print(f"Not zh data count: {count}")
print("Remove Emoji...")
for article in tqdm(data):
    emoji.replace_emoji(article['maintext'], replace='')
    emoji.replace_emoji(article['title'], replace='')

# 以下由ChatGPT協助產生，部分改寫

print("Remove 【】 and first()...")
for article in tqdm(data):
    article['maintext'] = re.sub(r"【.*?】", "", article['maintext'])
    article['maintext'] = re.sub(r"^\([^)]*\)\s*", "", article['maintext'])

print("Remove 《...》 in the end of the article...")
for article in tqdm(data):
    article['maintext'] = re.sub(r"(?<!^)(《.*?》)\s*$", "", article['maintext']).rstrip()

print("Remove （...） in the start of the article...")
for article in tqdm(data):
    article['maintext'] = re.sub(r"^\（.*?\）\s*", "", article['maintext']).rstrip()

print("Remove 圖/、文/ etc...")
for article in tqdm(data):
    article['maintext'] = re.sub(r"圖[．／/‧︱].*?(\n|$)", "", article['maintext'])
    article['maintext'] = re.sub(r"圖 /\s*.*?(\n|$)", "", article['maintext'])
    article['maintext'] = re.sub(r"文[．／/‧︱].*?(\n|$)", "", article['maintext'])
    article['maintext'] = re.sub(r"文 /\s*.*?(\n|$)", "", article['maintext'])
    article['maintext'] = re.sub(r"攝影[．／/‧︱].*?(\n|$)", "", article['maintext'])
    article['maintext'] = re.sub(r".*?／攝影.*?(?:。|\n)", "", article['maintext'])

print("Remove 「...轉載...」...")
for article in tqdm(data):
    article['maintext'] = re.sub(r"[^。\n]*轉載.*?(\n|$)", "", article['maintext'])

print("Remove 「...德國之聲版權...」...")
for article in tqdm(data):
    article['maintext'] = re.sub(r"[^。\n]*德國之聲版權.*?(\n|$)", "", article['maintext'])

print("Remove 「...圖片來源...」 「image source」...")
for article in tqdm(data):
    article['maintext'] = re.sub(r"[^。\n]*圖片來源.*?(\n|$)", "", article['maintext'])
    article['maintext'] = re.sub(r"[^。\n]*image source.*?(\n|$)", "", article['maintext'])
    article['maintext'] = re.sub(r"[^。\n]*Image Source.*?(\n|$)", "", article['maintext'])

print("Remove 「延伸閱讀...」...")
for article in tqdm(data):
    article['maintext'] = re.sub(r"延伸閱讀.*$", "", article['maintext'],flags=re.DOTALL)


print("Remove 「...@...」...")
for article in tqdm(data):
    article['maintext'] = re.sub(r".*@.*?(\n|$)", "", article['maintext'])

print("Remove mutiple \\n...")
for article in tqdm(data):
    article['maintext'] = re.sub(r"\n+', '\n", "", article['maintext'])
    article['maintext'] = article['maintext'].strip()

# 有空行的直接移掉
print("Remove empty column")
for article in tqdm(data):
    if len(article['maintext']) != 0:
        final_data.append(article)
    else:
        remove_data.append(article)

if args.prompt == True:
    print("Add prompt...")
    for article in tqdm(data):
        article['maintext'] = "請根據以下新聞內容生成標題：" + article['maintext']

print(f"Final data size:{len(final_data)}")

random.seed(datetime.now().timestamp())
random.shuffle(final_data)

train_size = int(args.train_size * len(final_data))

train_data = final_data[:train_size]
valid_data = final_data[train_size:]

print(f"Final train data size:{len(train_data)}")
print(f"Final valid data size:{len(valid_data)}")
print("Output file...")
with open(args.output_train, 'w', encoding='utf-8') as f_train:
    json.dump(train_data, f_train, ensure_ascii=False, indent=4)

with open(args.output_valid, 'w', encoding='utf-8') as f_valid:
    json.dump(valid_data, f_valid, ensure_ascii=False, indent=4)

if args.output_remove is not None and len(remove_data) != 0:
    with open(args.output_remove, 'w', encoding='utf-8') as f_remove:
        json.dump(remove_data, f_remove, ensure_ascii=False, indent=4)
print("Finish. Have a nice day.")

