import json
import os
from datasets import load_dataset
from tqdm import tqdm


def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[√] 已保存 {len(data)} 条数据到 -> {filename}")


def process_squad(ds, limit):
    processed = []
    print(f"[-] 正在处理 SQuAD 数据 ({limit}条)...")
    for item in tqdm(ds):
        if len(processed) >= limit: break
        processed.append({
            "question": item['question'],
            "context": item['context'],
            "answers": item['answers']['text'],
            "source": "squad"
        })
    return processed


def process_hotpot(ds, limit):
    processed = []
    print(f"[-] 正在处理 HotpotQA 数据 ({limit}条)...")
    for item in tqdm(ds):
        if len(processed) >= limit: break

        # 拼接文档
        titles = item['context']['title']
        sentences = item['context']['sentences']
        full_context = ""
        for title, sents in zip(titles[:3], sentences[:3]):
            full_context += f"【Title: {title}】{''.join(sents)}\n"

        processed.append({
            "question": item['question'],
            "context": full_context.strip(),
            "answers": [item['answer']],
            "source": "hotpot"
        })
    return processed


if __name__ == "__main__":
    # === 配置区域 ===
    TRAIN_SIZE = 6000  # 你的训练集大小
    TEST_SIZE = 2000  # 你的测试集大小

    # 1. 下载 SQuAD
    print("\n>>> 正在准备 SQuAD 数据集...")
    # 1.1 从官方 train 下载训练数据
    ds_train = load_dataset("squad", split="train", streaming=True)
    data_train = process_squad(ds_train, TRAIN_SIZE)
    save_to_json(data_train, "squad_train.json")

    # 1.2 从官方 validation 下载测试数据
    ds_test = load_dataset("squad", split="validation", streaming=True)
    data_test = process_squad(ds_test, TEST_SIZE)
    save_to_json(data_test, "squad_test.json")

    # 2. 下载 HotpotQA
    print("\n>>> 正在准备 HotpotQA 数据集...")
    # 2.1 从官方 train 下载训练数据
    ds_train = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
    data_train = process_hotpot(ds_train, TRAIN_SIZE)
    save_to_json(data_train, "hotpot_train.json")

    # 2.2 从官方 validation 下载测试数据
    ds_test = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    data_test = process_hotpot(ds_test, TEST_SIZE)
    save_to_json(data_test, "hotpot_test.json")

    print("\n" + "=" * 50)
    print("完美！数据分割完毕：")
    print(f"训练集 (用于训练路由): squad_train.json, hotpot_train.json (各 {TRAIN_SIZE} 条)")
    print(f"测试集 (用于最终跑分): squad_test.json, hotpot_test.json (各 {TEST_SIZE} 条)")
    print("=" * 50)