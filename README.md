# 2024ADL Homework2
## 環境與套件
程式語言：Python 3.8.20<br>
套件：PyTorch 2.1.0 transformers 4.44.2, datasets 2.19.1, accelerate 0.34.2 evaluate 0.4.0 tqdm 4.66.5 numpy 1.24.3 pandas 2.0.3 jsonlines 2.0.0<br>
另外Preprocess.py包含: langid 1.1.6 emoji 2.7.0<br>
作業系統：Windows 11 64bit
## 使用方式
### 1. 將jsonl轉成json
```
python ./Code/Unicodetransfer.py --input path/to/input --output path/to/output
```
### 2. 訓練(以表現最好的模型為例子)
```
python ./Code/run_summarization_no_trainer.py --train_file path/to/train_file --validation_file path/to/validation_file --model_name_or_path "google/mt5-small" --text_column "maintext" --summary_column "title" --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --learning_rate 3e-4 --gradient_accumulation_steps 2 --output_dir path/to/output --num_train_epochs 15 --num_beams 4 --max_source_length 256 --max_target_length 64
```
### 3. 預測
```
python ./Code/Evaluate.py --test_file path/to/test_file --output path/to/output --model_path path/to/Model --text_column "maintext" --summary_column "title" --max_source_length 256 --max_target_length 64 --num_beams 8
```
## download.sh & run.sh:
讓助教測試用的，download.sh會下載模型，run.sh會跑預測<br>
其中run.sh需要兩個參數:<br>
```
"${1}": path to input file.
"${2}": path to output file.
```
## Reference:
此作業是基於以下GitHub repo改寫：<br>
https://github.com/huggingface/transformers/tree/main
