#!/bin/bash
if [ $# -ne 2 ]; then
    echo "Error: run.sh need two arguments."
    echo "Usage: run.sh <path to input> <path to output>"
    exit 1
fi
echo "path to input: ${1}"
echo "path to output: ${2}"

input_file=$1
output_file=$2

echo "Transfer data to unicode."
python ./Code/Unicodetransfer.py --input "$input_file" --output ./testinput.json
echo "Running summarization."
python ./Code/Evaluate.py --test_file ./testinput.json --output "$output_file" --model_path ./Model_SU_final --text_column "maintext" --summary_column "title" --max_source_length 256 --max_target_length 64 --num_beams 8