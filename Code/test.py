# coding=utf-8
import re
text ="值得注意的是，在我們查資料的時候，發現2015年幾乎是最集中抗議基地台問題的一年，光是自由時報這一年，就至少有五篇以上的報導：原因在於2015年是當時4G上路後的第一年，正是各家業者拼基地台搶地盤的時候，因此這種狀況也特別多。同樣的我們也可以推估，今年業者號稱會5G上路，明年預料將會是這些基地台掮客大顯身手的一年，類似的新聞恐怕也少不了。《原文刊登於合作媒體T客邦，聯合新聞網獲授權轉載。》"
cleaned_text = re.sub(r"^.*※.*$", "", text, flags=re.MULTILINE)
cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
cleaned_text = re.sub(r'《.*?》\s*$', '', text).rstrip()
cleaned_text = re.sub(r'.*@.*?(\n|$)', '', text)
cleaned_text = re.sub(r'圖[．／/‧].*?(\n|$)', '', text)
cleaned_text = re.sub(r'文[．／/‧].*?(\n|$)', '', text)
cleaned_text = cleaned_text.strip()
print(cleaned_text)

text = "《這是需要刪除的部分》這是一個示例句子。《這是需要刪除的部分》"
# 只移除最後的「《...》」
cleaned_text = re.sub(r'(?<!^)(《.*?》)\s*$', '', text).rstrip()

print(cleaned_text)  # 輸出: "這是一個示例句子"