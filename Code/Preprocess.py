# 標題有/的先不管，感覺應該不影響
import argparse
import json, random
from datetime import datetime
import langid
import emoji
from tqdm import tqdm


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
args = parser.parse_args()
with open(args.input, 'r', encoding='utf-8') as f:
    rawdata = json.load(f)

data = []
remove_data = []
count = 0
for article in tqdm(rawdata):
    language, score = langid.classify(article['maintext'])
    if language != "zh":
        remove_data.append(article)
        count += 1
    else:
        data.append(article)

print(f"Not zh data count: {count}")
for article in tqdm(data):
    emoji.replace_emoji(article['maintext'], replace='')
    emoji.replace_emoji(article['title'], replace='')

# 以下部分由ChatGPT協助產生，部分改寫
random.seed(datetime.now().timestamp())
random.shuffle(data)

train_size = int(args.train_size * len(data))

train_data = data[:train_size]
valid_data = data[train_size:]

with open(args.output_train, 'w', encoding='utf-8') as f_train:
    json.dump(train_data, f_train, ensure_ascii=False, indent=4)

with open(args.output_valid, 'w', encoding='utf-8') as f_valid:
    json.dump(valid_data, f_valid, ensure_ascii=False, indent=4)

if args.output_remove is not None and len(remove_data) != 0:
    with open(args.output_remove, 'w', encoding='utf-8') as f_remove:
        json.dump(remove_data, f_remove, ensure_ascii=False, indent=4)

