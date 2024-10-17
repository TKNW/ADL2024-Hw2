#!/bin/bash
echo "Download file"
gdown --fuzzy "https://drive.google.com/file/d/1jVWJNy93aKpGyWRECmy7EK32RcXHZ6FH/view?usp=sharing"
echo "Unzip file"
unzip Model_SU_final.zip -d ./Model_SU_final
rm Model_SU_final.zip
echo "Download punkt_tab in nltk"
python <<EOF
import nltk
nltk.download('punkt_tab')
EOF