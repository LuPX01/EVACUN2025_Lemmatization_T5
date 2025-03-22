from transformers import ByT5Tokenizer, T5ForConditionalGeneration

# 加载微调后的模型和分词器
print("Loading model and tokenizer...")
tokenizer = ByT5Tokenizer.from_pretrained("../../byt5_small-ALL-lemma-v4")
model = T5ForConditionalGeneration.from_pretrained("../../byt5_small-ALL-lemma-v4")
print("Tokenizer vocab size:", len(tokenizer))
print("Tokenizer special tokens:", tokenizer.special_tokens_map)

# 将模型切换为评估模式
model.eval()

while True:
    # 提示用户输入
    input_word = input("Enter a word to convert or 'exit' to quit: ")

    if input_word.lower() == 'exit':
        print("Exiting the program.")
        break

    try:
        # 将输入格式化为模型需要的格式
        input_text = f"Convert: {input_word}"

        # 编码输入
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        # 生成输出
        outputs = model.generate(input_ids, max_length=128)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Predicted Lemma: {decoded_output}")
    except Exception as e:
        print(f"An error occurred: {e}")
