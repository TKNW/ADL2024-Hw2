import json
import argparse

parser = argparse.ArgumentParser(description="Transfer jsonl to Unicode json. For read easily.")
parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file.",
    )
parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file.",
    )
args = parser.parse_args()
jsonfile = args.input
outputfile = args.output
jsonlist = []
with open(jsonfile, 'r', encoding='utf-8') as f:
    for line in f:
        jsonobj = json.loads(line)
        jsonlist.append(jsonobj)
with open(outputfile, 'w', encoding='utf-8') as f:
    json.dump(jsonlist, f, ensure_ascii=False, indent=4)