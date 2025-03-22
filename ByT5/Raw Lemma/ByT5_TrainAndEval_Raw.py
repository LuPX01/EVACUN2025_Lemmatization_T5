# byT5_small
# 发现padding改成-100的确效果更佳

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import ByT5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from datasets import Dataset as HFDataset
import logging

# 确保保存文件的目录存在
log_dir = "byt5_lemma_prediction_logs"
os.makedirs(log_dir, exist_ok=True)

# 设置日志文件
logging.basicConfig(
    filename=os.path.join(log_dir, "training_log.txt"),
    filemode="w",
    format="%(asctime)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 自定义日志回调类
class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step:  # 如果有日志数据且有全局步数
            # logger.info(f"Step {state.global_step} - Training Loss: {logs.get('loss', 'N/A')}, Validation Loss: {logs.get('eval_loss', 'N/A')}")
            logger.info(f"Step {state.global_step} - "
                        f"Training Loss: {logs.get('loss', 'N/A')}, "
                        f"Validation Loss: {logs.get('eval_loss', 'N/A')}, "
                        f"Learning Rate: {logs.get('learning_rate', 'N/A')}")

# Step 1: 加载数据
logger.info("Loading data from Excel file.")
df = pd.read_excel("../../data/All_lemma_try_2_new.xlsx")

# 数据格式准备：将原词和lemma转换成合适的训练对
logger.info("Preparing data for training.")
df['input_text'] = df['clean_value'].apply(lambda x: f"convert: {x}")
df['target_text'] = df['lemma']
# df['target_text'] = df['lemma'].apply(lambda x: x.split()[0])  # 只保留lemma中的实际部分，去掉 "I" 等附加标记

# df['input_text'] = df['clean_value'].apply(lambda x: f"convert: {x}")
# df['target_text'] = df['clean_lemma']

print(df.head())  # 控制台显示前五行
df.to_excel(os.path.join(log_dir, "processed_data.xlsx"), index=False)
logger.info("Processed dataframe saved to Excel for inspection.")

# Step 2: 划分训练和验证集
logger.info("Splitting data into training and validation sets.")
train_data, val_data = train_test_split(df, test_size=0.05, random_state=42)

# # 保存训练和验证数据集到Excel文件，便于查看
# train_data.to_excel(os.path.join(log_dir, "LP_train_data.xlsx"), index=False)
# val_data.to_excel(os.path.join(log_dir, "LP_val_data.xlsx"), index=False)
# logger.info("Training and validation datasets have been saved as Excel files.")

# Step 3: 转换为 Hugging Face 数据格式
logger.info("Converting data to Hugging Face Dataset format.")
train_dataset = HFDataset.from_pandas(train_data[['input_text', 'target_text']])
val_dataset = HFDataset.from_pandas(val_data[['input_text', 'target_text']])

# 打印并记录 train_dataset 和 val_dataset 的前几行
logger.info(f"First few rows of train_dataset:\n{train_dataset[:5]}")
logger.info(f"First few rows of val_dataset:\n{val_dataset[:5]}")

# 记录数据集的列名
logger.info(f"Train dataset columns: {train_dataset.column_names}")
logger.info(f"Validation dataset columns: {val_dataset.column_names}")

# 记录数据集的大小
logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Validation dataset size: {len(val_dataset)}")

# 查看数据集中的一个样本
logger.info(f"Sample from train dataset: {train_dataset[0]}")
logger.info(f"Sample from val dataset: {val_dataset[0]}")

# Step 4: 加载 ByT5 Tokenizer 和模型
logger.info("Loading ByT5-small tokenizer and model.")
tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")

# 重新加载模型，避免缓存
def reload_model():
    model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
    model.train()  # 确保模型在训练模式
    return model

# model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

# # 数据预处理
# def preprocess_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# 数据预处理
def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']

    # Tokenize the inputs (with padding and truncation)
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)

    # Tokenize the targets (with padding and truncation)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)

    # Ensure labels are properly set as input_ids
    model_inputs["labels"] = labels["input_ids"]

    # Ensure padding is properly done, avoid list type issues
    print(f"Labels before replacement: {model_inputs['labels'][:2]}")
    # 是嵌套列表，遍历嵌套的每个列表进行替换
    model_inputs["labels"] = [
        [label if label != tokenizer.pad_token_id else -100 for label in labels]
        for labels in model_inputs["labels"]
    ]
    print(f"Labels after replacement: {model_inputs['labels'][:2]}")

    return model_inputs

# Step 5: 应用预处理
logger.info("Tokenizing training and validation data.")
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Step 6: 训练参数
logger.info("Setting up training arguments.")
training_args = TrainingArguments(
    output_dir="/Volumes/T7 Data/EVACUN_trained_model/byt5_small-ALL-lemma-v4.1",
    save_strategy="steps",  # 每隔一定步数进行保存
    # save_strategy="epoch",
    evaluation_strategy="steps",  # 每隔一定步数进行评估
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='/Volumes/T7 Data/EVACUN_trained_model/byt5_small-ALL-lemma-v4.1/logs',
    logging_steps=100,
    save_total_limit=3,
    save_steps=1000,
    eval_steps=1000,
    load_best_model_at_end=True,
    logging_first_step=True,  # 记录第一个训练步骤的日志
    # 配置早停
    metric_for_best_model = "eval_loss",  # 选择监控的指标
    greater_is_better = False,  # eval_loss 越低越好
)

# 重新加载模型，避免缓存
model = reload_model()

# 添加早停回调，并设置耐心值为2
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2  # 设置耐心值为2步
)

# Step 7: 定义 Trainer
logger.info("Initializing Trainer.")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    callbacks=[LogCallback(), early_stopping_callback]  # 使用自定义的回调类
)

# Step 8: 开始训练，微调模型
logger.info("Starting training.")
trainer.train()

# Step 9: 保存模型和分词器
logger.info("Saving model and tokenizer.")
model.save_pretrained("../../byt5_small-ALL-lemma-v4.1")
tokenizer.save_pretrained("../../byt5_small-ALL-lemma-v4.1")

logger.info("Training completed and artifacts saved.")
