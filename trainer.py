# trainer/trainer.py 最终完整版 (包含分类权重调整 + StepLR平缓调度 + 多维指标)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

'''
#当前防过拟合与对齐策略：
1. L2正则化：去掉
2. 高斯噪声数据增强：保留（Jittering 增加鲁棒性）
3. 数据分类权重调整：恢复（解决 7 分类数据集严重的样本不平衡问题）
4. 学习率调度：恢复（StepLR step_size=20, 兼顾轻量化与巨型网络）
'''


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model.to(config['device'])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config['device']

        # 数据分类权重调整，解决类别不平衡问题 (config['class_weights']: list / 'auto' / None)
        weights = self._build_class_weights(config.get('class_weights'), config['num_classes'])
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # 仅使用 Adam 优化器，移除多余的 L2 正则化
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        # 平缓的学习率衰减 (默认每 20 轮衰减一半)
        self.lr_step = config.get('lr_step', 20)
        self.lr_gamma = config.get('lr_gamma', 0.5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step, gamma=self.lr_gamma)

        self.noise_std = config.get('noise_std', 0.02)

        self.best_acc = 0.0
        self.best_val_loss = float('inf')
        self.early_stop_patience = config.get('early_stop_patience', 10)
        self.early_stop_counter = 0
        self.early_stop = False

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # 获取模型名称并自动生成专属文件夹
        self.model_name = self.model.__class__.__name__
        self.save_dir = os.path.join(config['save_dir'], self.model_name)
        self.log_dir = os.path.join(config['log_dir'], self.model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _build_class_weights(self, spec, num_classes):
        if spec is None:
            return None
        if spec == 'auto':
            # 逆频率权重，按类别数归一化使均值为 1
            labels = torch.as_tensor(self.train_loader.dataset.y)
            counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1)
            w = counts.sum() / (num_classes * counts)
        else:
            w = torch.tensor(spec, dtype=torch.float32)
            if len(w) != num_classes:
                raise ValueError(f"class_weights has {len(w)} entries, expected {num_classes}")
        return w.to(self.device)

    def calculate_model_size(self):
        """计算模型的参数量和体积（MB）"""
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        size_mb = total_params * 4 / (1024 ** 2)
        return total_params, size_mb

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        preds_all, labels_all = [], []

        for x, y in tqdm(self.train_loader, desc=f"Training {self.model_name}"):
            x, y = x.to(self.device), y.to(self.device)

            # 数据增强 (Jittering)。给 CSI 信号加入微小的动态高斯噪声
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(y.cpu().numpy())

        acc = accuracy_score(labels_all, preds_all)
        return total_loss / len(self.train_loader), acc

    def val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        preds_all, labels_all = [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                total_loss += loss.item()
                pred = torch.argmax(out, dim=1)
                preds_all.extend(pred.cpu().numpy())
                labels_all.extend(y.cpu().numpy())

        acc = accuracy_score(labels_all, preds_all)
        return total_loss / len(self.val_loader), acc

    def run(self):
        total_params, size_mb = self.calculate_model_size()
        print(f"\n🚀 Start Training {self.model_name} (LR Scheduler: StepLR({self.lr_step}, {self.lr_gamma}), "
              f"Class Weights: {'On' if self.criterion.weight is not None else 'Off'})...")
        print(f"📦 Model Parameters: {total_params:,} | Size: {size_mb:.4f} MB\n")

        for epoch in range(self.config['epochs']):
            if self.early_stop:
                print("\n🛑 Early Stopping triggered!")
                break

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.val_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # 获取当前的学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"[Epoch {epoch + 1:2d}] LR: {current_lr:.6f} | "
                  f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
                  f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

            # 调度器更新
            self.scheduler.step()

            # 基于 val_acc 的保存与早停逻辑
            if val_acc >= self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best.pth"))
                self.early_stop_counter = 0  # 只要准确率创新高，就重置耐心值
                print(f"✅ Best model saved (Val Acc: {val_acc:.4f})")
            else:
                self.early_stop_counter += 1
                print(f"⚠️ Early stop counter: {self.early_stop_counter}/{self.early_stop_patience}")
                if self.early_stop_counter >= self.early_stop_patience:
                    self.early_stop = True

        try:
            self.plot_curve()
        except Exception as e:
            print(f"⚠️ Plotting skipped (matplotlib error: {e})")

    def test(self, class_names):
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, "best.pth")))
        self.model.eval()
        preds_all, labels_all = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                out = self.model(x)
                pred = torch.argmax(out, dim=1)
                preds_all.extend(pred.cpu().numpy())
                # 确保 y 也被转换为 numpy，防止设备报错
                labels_all.extend(y.cpu().numpy() if hasattr(y, 'cpu') else y.numpy())

        # 1. 计算论文要求的四个核心性能指标 (使用 macro 平均)
        acc = accuracy_score(labels_all, preds_all)
        precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='macro')

        # 2. 获取参数量与体积
        total_params, size_mb = self.calculate_model_size()

        # 3. 生成详细分类报告
        report = classification_report(labels_all, preds_all, target_names=class_names, digits=4)

        # 4. 将所有指标和参数量写入测试结果文件
        test_results_text = (
            f"=== Model Performance Metrics ===\n"
            f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)\n"
            f"Precision: {precision:.4f} ({precision * 100:.2f}%)\n"
            f"Recall:    {recall:.4f} ({recall * 100:.2f}%)\n"
            f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)\n\n"
            f"=== Model Complexity ===\n"
            f"Total Parameters: {total_params:,}\n"
            f"Model Size:       {size_mb:.4f} MB\n"
        )

        with open(os.path.join(self.log_dir, "test_metrics_and_size.txt"), "w", encoding="utf-8") as f:
            f.write(test_results_text)
            f.write("\n=== Detailed Classification Report ===\n")
            f.write(report)

        # 5. 在终端打印结果
        print(f"\n🎯 {self.model_name} Test Results:")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   Parameters: {total_params:,} ({size_mb:.4f} MB)\n")
        print(report)

        try:
            self.plot_cm(labels_all, preds_all, class_names)
        except Exception as e:
            print(f"⚠️ Confusion matrix skipped (Error: {e})")

        # 返回一个包含所有指标的字典，方便 run_all.py 提取汇总
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "params": total_params,
            "size_mb": size_mb
        }

    def plot_curve(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.val_losses, label='val loss')
        plt.title(f"{self.model_name} Loss Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='train acc')
        plt.plot(self.val_accs, label='val acc')
        plt.title(f"{self.model_name} Accuracy Curve")
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "curve.png"))
        plt.close()

    def plot_cm(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{self.model_name} Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.log_dir, "cm.png"))
        plt.close()