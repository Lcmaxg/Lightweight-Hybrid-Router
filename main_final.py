import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rank_bm25 import BM25Okapi
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ==========================================
# 0. 科研标配：固定随机种子
# ==========================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)


# ==========================================
# 1. 模型定义：跨模态注意力路由 (Attention Router)
# ==========================================
class AttentionHybridRouter(nn.Module):
    def __init__(self, stat_dim=4, emb_dim=384):
        super(AttentionHybridRouter, self).__init__()

        # --- A. 语义编码器 (Semantic Context) ---
        # 作用：理解问题意图 (e.g., "这是个多跳推理题")
        self.sem_encoder = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # --- B. 跨模态注意力门控 (Attention Gate) ---
        # 核心创新：根据语义上下文，决定关注哪个统计特征
        # Input: 64 (语义) -> Output: 4 (统计特征的权重)
        self.attn_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, stat_dim),
            nn.Softmax(dim=1)  # 归一化，保证权重和为1
        )

        # --- C. 统计特征编码器 ---
        self.stat_encoder = nn.Sequential(
            nn.Linear(stat_dim, 16),
            nn.ReLU()
        )

        # --- D. 最终决策层 ---
        self.fc_final = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 0=Fast, 1=Slow
        )

    def forward(self, stats, embedding):
        # 1. 获取语义上下文
        sem_ctx = self.sem_encoder(embedding)  # [batch, 64]

        # 2. 生成注意力权重
        attn_weights = self.attn_gate(sem_ctx)  # [batch, 4]

        # 3. 特征重加权 (Feature Reweighting)
        # 显式注意力：让模型动态放大或缩小某些统计指标
        stats_weighted = stats * attn_weights

        # 4. 编码加权后的统计特征
        stat_ctx = self.stat_encoder(stats_weighted)  # [batch, 16]

        # 5. 拼接与分类
        combined = torch.cat((sem_ctx, stat_ctx), dim=1)
        logits = self.fc_final(combined)

        return logits


# ==========================================
# 2. 实验主逻辑
# ==========================================
class FinalExperiment:
    def __init__(self, model_name='BAAI/bge-small-en-v1.5'):
        print(f"\n[1/6] 初始化环境 (Model: {model_name})...")
        self.train_data = self._load_data('squad_train.json', 'hotpot_train.json')
        self.test_data = self._load_data('squad_test.json', 'hotpot_test.json')

        # 构建知识库
        print(f"    - 构建知识库索引 (文档总数: {len(self.train_data) + len(self.test_data)})...")
        self.corpus = list(set([d['context'] for d in self.train_data + self.test_data]))
        self.bm25 = BM25Okapi([doc.split(" ") for doc in self.corpus])

        # 向量模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"    - 加载 SOTA 模型到 {self.device}...")
        self.encoder = SentenceTransformer(model_name)
        self.encoder.to(self.device)

        # 预计算知识库向量
        print("    - 向量化知识库 (Pre-computing Corpus Embeddings)...")
        # 检查是否有本地缓存，避免重复计算
        if os.path.exists("corpus_embeddings.pt"):
            self.corpus_embeddings = torch.load("corpus_embeddings.pt", map_location=self.device)
        else:
            self.corpus_embeddings = self.encoder.encode(
                self.corpus,
                convert_to_tensor=True,
                device=self.device,
                batch_size=128,
                show_progress_bar=True
            )
            torch.save(self.corpus_embeddings, "corpus_embeddings.pt")

    def _load_data(self, f1, f2):
        try:
            with open(f1, 'r') as fa, open(f2, 'r') as fb:
                return json.load(fa) + json.load(fb)
        except:
            print("❌ 数据文件缺失！请先运行 download_data.py")
            exit()

    def extract_features_single(self, query):
        """实时提取单条特征"""
        q_emb_tensor = self.encoder.encode(query, convert_to_tensor=True, device=self.device)

        hits = util.semantic_search(q_emb_tensor, self.corpus_embeddings, top_k=2)[0]
        score1 = hits[0]['score']
        score2 = hits[1]['score'] if len(hits) > 1 else 0
        gap = score1 - score2

        bm25_scores = self.bm25.get_scores(query.split(" "))
        bm25_top1 = np.max(bm25_scores)

        stats = np.array([bm25_top1 / 50.0, score1, gap, len(query) / 100.0], dtype=np.float32)
        return stats, q_emb_tensor.cpu().numpy(), hits, bm25_scores

    def prepare_data(self, raw_data, cache_file):
        """生成并缓存数据"""
        if os.path.exists(cache_file):
            print(f"\n[Cache] ⚡️ 秒级加载缓存: {cache_file}")
            data = np.load(cache_file)
            return data['stats'], data['emb'], data['label']

        print(f"\n[Gen] 正在生成特征并标注 Oracle (存入 {cache_file})...")
        X_stats, X_emb, Y = [], [], []

        for item in tqdm(raw_data):
            query = item['question']
            truth = item['context']
            stats, emb, hits, bm25_scores = self.extract_features_single(query)

            # Oracle 逻辑
            bm25_doc = self.corpus[np.argmax(bm25_scores)]
            label = 0 if bm25_doc == truth else 1

            X_stats.append(stats)
            X_emb.append(emb)
            Y.append(label)

        np.savez_compressed(cache_file, stats=X_stats, emb=X_emb, label=Y)
        return np.array(X_stats), np.array(X_emb), np.array(Y)

    def train(self, X_s, X_e, Y):
        print("\n[3/6] 训练 Attention Hybrid Router...")

        indices = np.arange(len(Y))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        train_set = TensorDataset(
            torch.tensor(X_s[train_idx]).to(self.device),
            torch.tensor(X_e[train_idx]).to(self.device),
            torch.tensor(Y[train_idx]).long().to(self.device)
        )
        val_set = TensorDataset(
            torch.tensor(X_s[val_idx]).to(self.device),
            torch.tensor(X_e[val_idx]).to(self.device),
            torch.tensor(Y[val_idx]).long().to(self.device)
        )

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

        # === 实例化 Attention 模型 ===
        model = AttentionHybridRouter().to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        best_acc = 0.0
        # === 修改：Epoch 增加到 25 ===
        epochs = 25

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for b_s, b_e, b_y in train_loader:
                optimizer.zero_grad()
                out = model(b_s, b_e)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            correct = 0;
            total = 0;
            val_loss = 0
            with torch.no_grad():
                for b_s, b_e, b_y in val_loader:
                    out = model(b_s, b_e)
                    loss = criterion(out, b_y)
                    val_loss += loss.item()
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == b_y).sum().item()
                    total += b_y.size(0)

            val_acc = correct / total
            scheduler.step(val_loss)

            print(f"    Epoch {epoch + 1:02d} | Loss: {train_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_attn_model.pth")

        print(f"    [√] 训练完成. Best Val Acc: {best_acc:.4f}")
        model.load_state_dict(torch.load("best_attn_model.pth"))
        return model

    def evaluate_benchmark_real_time(self, model, raw_test_data):
        print("\n[4/6] 正在进行真实性能压测 (Real-Time Benchmark)...")
        model.eval()

        metrics = {"Naive": 0, "Baseline": 0, "Ours": 0, "Oracle": 0}
        times = {"Naive": 0, "Baseline": 0, "Ours": 0}
        all_oracle, all_preds = [], []
        router_stats = {"Fast": 0, "Slow": 0}

        print("    -> 正在逐条回放测试集...")
        for item in tqdm(raw_test_data):
            query = item['question']
            truth = item['context']

            # 1. Naive (BM25)
            t0 = time.perf_counter()
            bm25_scores = self.bm25.get_scores(query.split(" "))
            bm25_doc = self.corpus[np.argmax(bm25_scores)]
            t_naive = time.perf_counter() - t0
            times["Naive"] += t_naive
            if bm25_doc == truth: metrics["Naive"] += 1

            # 2. Baseline (BGE)
            t0 = time.perf_counter()
            q_emb = self.encoder.encode(query, convert_to_tensor=True, device=self.device)
            hits = util.semantic_search(q_emb, self.corpus_embeddings, top_k=2)[0]
            vec_doc = self.corpus[hits[0]['corpus_id']]
            t_base = time.perf_counter() - t0
            times["Baseline"] += t_base
            if vec_doc == truth: metrics["Baseline"] += 1

            # 3. Ours (Attention)
            t0 = time.perf_counter()
            # 准备输入 (模拟共享计算)
            gap = hits[0]['score'] - (hits[1]['score'] if len(hits) > 1 else 0)
            bm25_max = np.max(bm25_scores)
            stats = torch.tensor([bm25_max / 50.0, hits[0]['score'], gap, len(query) / 100.0]).float().unsqueeze(0).to(
                self.device)
            emb_in = q_emb.unsqueeze(0)

            with torch.no_grad():
                logits = model(stats, emb_in)
                decision = torch.argmax(logits, dim=1).item()

            # 记录 F1
            oracle_label = 0 if bm25_doc == truth else 1
            all_oracle.append(oracle_label)
            all_preds.append(decision)

            # 路由选择
            t_router = time.perf_counter() - t0
            # Ours Time = Baseline Time (Encoding) + Router Time
            # 注意: 如果选BM25，理论上可以省去ANN Search时间，但必须保留Encoding时间
            times["Ours"] += (t_base + t_router)

            if decision == 0:
                router_stats["Fast"] += 1
                if bm25_doc == truth: metrics["Ours"] += 1
            else:
                router_stats["Slow"] += 1
                if vec_doc == truth: metrics["Ours"] += 1

            if (bm25_doc == truth) or (vec_doc == truth): metrics["Oracle"] += 1

        n = len(raw_test_data)
        final_acc = {k: v / n for k, v in metrics.items()}
        final_time = {k: (v / n) * 1000 for k, v in times.items()}

        print("\n" + "=" * 60)
        print("          ROUTER CLASSIFICATION REPORT (F1 Score)")
        print("=" * 60)
        print(classification_report(all_oracle, all_preds, target_names=["Fast (BM25)", "Slow (Vector)"], digits=4))

        return final_acc, final_time, router_stats


if __name__ == "__main__":
    exp = FinalExperiment()

    # 1. 准备数据
    Xs_train, Xe_train, Y_train = exp.prepare_data(exp.train_data, "train_data_final.npz")

    # 2. 训练 (25 Epochs)
    model = exp.train(Xs_train, Xe_train, Y_train)

    # 3. 评估
    acc, lat, stats = exp.evaluate_benchmark_real_time(model, exp.test_data)

    # 4. 打印报表
    print("\n" + "=" * 70)
    print("             FINAL EXPERIMENT RESULTS (Attention-Based)")
    print("=" * 70)
    print(f"{'Method':<20} | {'Hit Rate':<10} | {'Real Latency':<15} | {'vs. Baseline'}")
    print("-" * 65)
    print(
        f"{'Naive (BM25)':<20} | {acc['Naive']:.2%}     | {lat['Naive']:.2f} ms        | {lat['Baseline'] / lat['Naive']:.1f}x Faster")
    print(f"{'Baseline (BGE)':<20} | {acc['Baseline']:.2%}     | {lat['Baseline']:.2f} ms        | 1.0x (Ref)")
    print(
        f"{'Ours (Hybrid)':<20} | {acc['Ours']:.2%}     | {lat['Ours']:.2f} ms        | {lat['Baseline'] / lat['Ours']:.2f}x Speedup")
    print(f"{'Oracle (Ideal)':<20} | {acc['Oracle']:.2%}     | -                | -")
    print("-" * 65)
    fast_ratio = stats['Fast'] / (stats['Fast'] + stats['Slow'])
    print(f"Router Decisions: Fast (BM25) = {fast_ratio:.1%} | Slow (Vector) = {1 - fast_ratio:.1%}")
    print("=" * 70)