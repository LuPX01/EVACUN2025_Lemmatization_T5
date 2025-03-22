from transformers import T5Tokenizer, MT5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

# 加载微调后的模型和分词器
tokenizer = T5Tokenizer.from_pretrained("../../mt5_small-ALL-lemma-v2")
model = MT5ForConditionalGeneration.from_pretrained("../../mt5_small-ALL-lemma-v2")

# 将模型切换为评估模式
model.eval()

# 读取数据集 Excel 文件
data_file_path = "../../data/All_lemma_try_2_new.xlsx"
df = pd.read_excel(data_file_path)

# 使用 train_test_split 分离测试集
data_train, data_test = train_test_split(df, test_size=0.05, random_state=42)

# 测试集的原词和正确的 lemma
original_words = data_test["clean_value"].tolist()
true_lemmas = data_test["lemma"].tolist()

# true_lemmas = [lemma.split()[0] for lemma in data_test["lemma"].tolist()]

# # 处理 lemma 列，去掉罗马数字部分，并返回处理后的文本作为 true_lemmas
# roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
# true_lemmas = data_test["lemma"].apply(lambda x: ' '.join(x.split()[:-1]) if x.split()[-1] in roman_numerals else x).tolist()

# 存储预测结果
predicted_lemmas = []
exact_match_flags = []

for word, true_lemma in zip(original_words, true_lemmas):
    try:
        # 将输入格式化为模型需要的格式
        input_text = f"Convert: {word}"

        # 编码输入
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        # 生成输出
        outputs = model.generate(input_ids, max_length=128)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 保存预测结果
        predicted_lemmas.append(decoded_output)
        exact_match_flags.append("√" if decoded_output == true_lemma else "×")
    except Exception as e:
        print(f"Error processing word '{word}': {e}")
        predicted_lemmas.append("ERROR")
        exact_match_flags.append("×")

# 过滤掉预测失败的条目，确保错误的预测不会影响分数计算
filtered_true_lemmas = [true for true, pred in zip(true_lemmas, predicted_lemmas) if pred != "ERROR"]
filtered_predicted_lemmas = [pred for pred in predicted_lemmas if pred != "ERROR"]

# 计算 Exact Match (EM)
correct_matches = sum([1 for true, pred in zip(filtered_true_lemmas, filtered_predicted_lemmas) if true == pred])
total_predictions = len(filtered_true_lemmas)
exact_match = correct_matches / total_predictions if total_predictions > 0 else 0

# 计算 BLEU 分数
smoothing_function = SmoothingFunction().method1

bleu_scores = []
for true, pred in zip(filtered_true_lemmas, filtered_predicted_lemmas):
    reference = [list(true)]  # 将正确的 lemma 转换为字符级列表
    hypothesis = list(pred)  # 将预测的 lemma 转换为字符级列表
    bleu = sentence_bleu(reference, hypothesis, weights=(1.0, 0, 0, 0), smoothing_function=smoothing_function)  # 使用 unigram BLEU
    bleu_scores.append(bleu)
average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

# 打印结果
print(f"Exact Match (EM): {exact_match:.4f}")
print(f"Average BLEU Score: {average_bleu:.4f}")

# 将预测结果和指标写入 Excel 文件
output_file_path = "../../results/mT5v2_ALL_predictions_with_metrics.xlsx"
data_test["predicted_lemma"] = predicted_lemmas
data_test["Exact Match"] = exact_match_flags

# 将 BLEU 分数作为列添加
bleu_column = bleu_scores + [None] * (len(data_test) - len(bleu_scores))  # 保持行数一致
data_test["BLEU Score"] = bleu_column

# 添加总体指标到单独的表格顶部
metrics_summary = pd.DataFrame({
    "clean_value": [""],
    "lemma": [""],
    "predicted_lemma": [""],
    "Exact Match": [f"Overall: {exact_match:.4f}"],
    "BLEU Score": [average_bleu]
})
data_test = pd.concat([metrics_summary, data_test], ignore_index=True)

# 保存到 Excel
data_test.to_excel(output_file_path, index=False)
print(f"Predictions and metrics saved to {output_file_path}")
