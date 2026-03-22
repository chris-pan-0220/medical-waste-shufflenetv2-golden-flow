#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import csv
import argparse
import copy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import QuantStub, DeQuantStub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# ----------------------------
# 1. 定義 Nano-ShuffleNetV2 (修正版)
# ----------------------------
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, dilation=1, groups=i, bias=bias)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

class NanoShuffleNetV2_10k(nn.Module):
    def __init__(self, num_classes=13):
        super(NanoShuffleNetV2_10k, self).__init__()
        
        # === 10k 版本配置 ===
        # 1. 移除 Conv5 (原本佔用大量參數)
        # 2. 將前面的通道數稍微加寬，把參數用在刀口上
        # 原本: [-1, 12, 16, 32, 48, 128]
        # 修改: [-1, 16, 24, 32, 64] -> 最後輸出 64 channel
        self.stage_out_channels = [-1, 16, 24, 32, 64]
        
        # 維持原本的深度，保證準確率
        self.stage_repeats = [2, 2, 2]
        
        self.quant = QuantStub()
        self.de_quant = DeQuantStub()

        input_channels = 3
        output_channels = self.stage_out_channels[1] # 16
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        
        self.stages = nn.ModuleList([])
        for name, repeats, output_channels in zip(
            ["Stage2", "Stage3", "Stage4"], 
            self.stage_repeats,
            self.stage_out_channels[2:]
        ):
            seq = []
            for i in range(repeats):
                if i == 0:
                    seq.append(InvertedResidual(input_channels, output_channels, 2))
                else:
                    seq.append(InvertedResidual(output_channels, output_channels, 1))
                input_channels = output_channels
            self.stages.append(nn.Sequential(*seq))

        # === 修改: 移除 Conv5 ===
        # 直接使用 Stage 4 的輸出 (64 ch) 接 Global Pool
        final_channels = self.stage_out_channels[-1]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        for stage in self.stages:
            x = stage(x)
        # x = self.conv5(x) # 已移除
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.de_quant(x)
        return x
        
    def fuse_model(self):
        from torch.ao.quantization import fuse_modules
        
        # 1. Fuse Conv1 (Conv + BN + ReLU)
        # 結構: [0:Conv, 1:BN, 2:ReLU]
        fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        
        for stage in self.stages:
            for block in stage:
                # 針對 Stride > 1 的情況 (有 branch1 和 branch2)
                if block.stride > 1:
                    # --- Branch 1 ---
                    # 結構: [0:DWConv, 1:BN, 2:PWConv, 3:BN, 4:ReLU]
                    
                    # Fuse 第一個 DWConv + BN (沒有 ReLU)
                    fuse_modules(block.branch1, ['0', '1'], inplace=True)
                    # Fuse 後面的 PWConv + BN + ReLU
                    fuse_modules(block.branch1, ['2', '3', '4'], inplace=True)
                    
                    # --- Branch 2 ---
                    # 結構: [0:PWConv, 1:BN, 2:ReLU, 3:DWConv, 4:BN, 5:PWConv, 6:BN, 7:ReLU]
                    
                    # Fuse 前段 PWConv + BN + ReLU
                    fuse_modules(block.branch2, ['0', '1', '2'], inplace=True)
                    # Fuse 中段 DWConv + BN (原本漏掉這裡！因為它沒有接 ReLU)
                    fuse_modules(block.branch2, ['3', '4'], inplace=True)
                    # Fuse 後段 PWConv + BN + ReLU
                    fuse_modules(block.branch2, ['5', '6', '7'], inplace=True)
                
                # 針對 Stride = 1 的情況 (只有 branch2)
                else:
                    # --- Branch 2 ---
                    # 結構同上
                    fuse_modules(block.branch2, ['0', '1', '2'], inplace=True)
                    fuse_modules(block.branch2, ['3', '4'], inplace=True) # 補上這裡
                    fuse_modules(block.branch2, ['5', '6', '7'], inplace=True)

# ----------------------------
# 2. 工具函數
# ----------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_transforms(img_size=128):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

def plot_history(history, out_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))
    
    # 1. Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.plot(epochs, history['meta_acc'], 'g--', label='Val Meta Acc')
    plt.title('Accuracy (Fine vs Meta)')
    plt.xlabel('Epochs')
    plt.ylabel('Acc (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_confusion_matrix_png(cm, classes, out_path, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ----------------------------
# 3. 訓練與評估核心
# ----------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, meta_mapping):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    meta_correct = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item() * inputs.size(0)

        with torch.no_grad():
            meta_pred = meta_mapping[predicted]
            meta_tgt = meta_mapping[labels]
            meta_correct += (meta_pred == meta_tgt).sum().item()

    avg_loss = total_loss / total
    acc = 100. * correct / total
    meta_acc = 100. * meta_correct / total
    return avg_loss, acc, meta_acc

def evaluate(model, loader, criterion, device, meta_mapping):
    model.eval()
    correct = 0
    total = 0
    meta_correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            meta_pred = meta_mapping[predicted]
            meta_tgt = meta_mapping[labels]
            meta_correct += (meta_pred == meta_tgt).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = 100. * correct / total
    meta_acc = 100. * meta_correct / total
    return acc, meta_acc, np.array(all_preds), np.array(all_targets)

# ----------------------------
# 4. 主程式
# ----------------------------

def main():
    # =============== 設定區 ===============
    # 請修改為您的資料集路徑
    data_root = r'C:\Users\USER\Desktop\EE_project\datasets\selfbuilt_dataset_integrate'
    output_dir = Path('output_nano_shufflenet_v2_10k') # 修改輸出資料夾名稱
    
    epochs_phase1 = 50 
    epochs_phase2 = 15  
    batch_size = 32
    lr_phase1 = 0.002
    lr_phase2 = 1e-4
    seed = 42
    # ======================================

    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data Loading
    print(f"Loading data from {data_root}...")
    train_tf, val_tf = get_transforms()
    full_ds = datasets.ImageFolder(root=data_root)
    class_names = full_ds.classes
    print(f"Detected {len(class_names)} Classes: {class_names}")

    # Meta Mapping
    meta_class_names = ["Infectious_waste", "Medicine_waste", "Sharps_waste", "Spare"]
    leaf_to_meta_idx = {
        "bandage": 0, "cotton_swab": 0, "gauze": 0, "gloves": 0, "masks": 0,
        "medicine": 1, "medicine_bottles": 1, "medicine_hospital_bag": 1, 
        "medicine_in_bag": 1, "pill_with_package": 1,
        "long_knife": 2, "short_knife": 2, "needle": 2,
        "spare": 3,
    }
    mapping_list = []
    for cname in class_names:
        if cname in leaf_to_meta_idx:
            mapping_list.append(leaf_to_meta_idx[cname])
        else:
            print(f"Warning: Class {cname} not in mapping! Defaulting to 0.")
            mapping_list.append(0)
    meta_mapping_tensor = torch.tensor(mapping_list, dtype=torch.long, device=device)
    meta_mapping_np = np.array(mapping_list)

    # Split
    from sklearn.model_selection import train_test_split
    targets = [s[1] for s in full_ds.samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=0.2, stratify=targets, random_state=seed
    )
    train_ds = datasets.ImageFolder(data_root, transform=train_tf)
    val_ds = datasets.ImageFolder(data_root, transform=val_tf)
    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Init Model (使用 PicoShuffleNetV2)
    model = NanoShuffleNetV2_10k(num_classes=len(class_names))
    model.to(device)
    print(f"Model Parameters: {count_parameters(model)}")

    criterion = nn.CrossEntropyLoss()
    
    # 記錄數據的字典
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_acc': [], 'meta_acc': [],
        'stage': [] 
    }

    # === Phase 1: Pre-training ===
    print(f"\n=== Phase 1: Pre-training ({epochs_phase1} epochs) ===")
    optimizer = optim.AdamW(model.parameters(), lr=lr_phase1, weight_decay=5e-5) # 降低 weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_phase1)

    for epoch in range(epochs_phase1):
        loss, acc, _ = train_one_epoch(model, train_loader, criterion, optimizer, device, meta_mapping_tensor)
        val_acc, val_meta_acc, _, _ = evaluate(model, val_loader, criterion, device, meta_mapping_tensor)
        scheduler.step()

        print(f"Ep {epoch+1}/{epochs_phase1} | Loss:{loss:.4f} | Train Acc:{acc:.2f}% | Val Acc:{val_acc:.2f}% | Meta Acc:{val_meta_acc:.2f}%")
        
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        history['val_acc'].append(val_acc)
        history['meta_acc'].append(val_meta_acc)
        history['stage'].append('Pretrain')
    
    torch.save(model.state_dict(), output_dir / 'pretrained_float.pt')

    # === Phase 2: QAT Fine-tuning ===
    print(f"\n=== Phase 2: QAT Fine-tuning ({epochs_phase2} epochs) ===")
    model.cpu()
    
    # A. Fuse (Eval mode)
    model.eval()
    model.fuse_model()
    
    # B. Config
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    
    # C. Train mode for prepare
    model.train()
    
    # D. Prepare
    torch.ao.quantization.prepare_qat(model, inplace=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr_phase2)
    best_acc = 0.0

    for epoch in range(epochs_phase2):
        loss, acc, _ = train_one_epoch(model, train_loader, criterion, optimizer, device, meta_mapping_tensor)
        val_acc, val_meta_acc, preds, targets = evaluate(model, val_loader, criterion, device, meta_mapping_tensor)

        print(f"QAT Ep {epoch+1}/{epochs_phase2} | Loss:{loss:.4f} | Val Acc:{val_acc:.2f}% | Meta Acc:{val_meta_acc:.2f}%")

        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        history['val_acc'].append(val_acc)
        history['meta_acc'].append(val_meta_acc)
        history['stage'].append('QAT')

        torch.save(model.state_dict(), output_dir / 'last.pt')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best.pt')

    # === Outputs ===
    print("\n=== Generating Final Outputs ===")
    model.load_state_dict(torch.load(output_dir / 'best.pt', map_location=device))
    final_acc, final_meta_acc, final_preds, final_targets = evaluate(model, val_loader, criterion, device, meta_mapping_tensor)
    
    model.cpu()
    model.eval()
    quantized_model = torch.ao.quantization.convert(model, inplace=False)
    torch.save(quantized_model.state_dict(), output_dir / 'nano_shufflenet_int8.pt')
    
    # 1. Plot
    plot_history(history, output_dir / 'training_history.png')
    
    # 2. CSV Log
    csv_path = output_dir / 'training_log.csv'
    print(f"Saving training log to {csv_path}...")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc', 'meta_acc', 'stage'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                history['train_loss'][i],
                history['train_acc'][i],
                history['val_acc'][i],
                history['meta_acc'][i],
                history['stage'][i]
            ])

    # 3. Confusion Matrices
    cm = confusion_matrix(final_targets, final_preds)
    save_confusion_matrix_png(cm, class_names, output_dir / 'confusion_matrix.png', title='Confusion Matrix (Sub-classes)')

    meta_preds = meta_mapping_np[final_preds]
    meta_targets = meta_mapping_np[final_targets]
    cm_meta = confusion_matrix(meta_targets, meta_preds)
    save_confusion_matrix_png(cm_meta, meta_class_names, output_dir / 'confusion_matrix_meta.png', title='Confusion Matrix (Meta-classes)')

    # 4. JSONs & Text
    with open(output_dir / 'labels.json', 'w', encoding='utf-8') as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    precision, recall, f1, _ = precision_recall_fscore_support(final_targets, final_preds, average=None)
    per_class_stats = []
    
    print(f"\n{'Class Name':<25} {'Prec':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        p, r, f = (precision[i], recall[i], f1[i]) if i < len(precision) else (0,0,0)
        print(f"{name:<25} {p:.3f}      {r:.3f}      {f:.3f}")
        per_class_stats.append({
            "class": name, "precision": float(p), "recall": float(r), "f1": float(f),
            "super_class": meta_class_names[mapping_list[i]]
        })

    with open(output_dir / 'per_class_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(per_class_stats, f, ensure_ascii=False, indent=2)

    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(final_targets, final_preds, average='macro')
    with open(output_dir / 'final_metrics_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Final Validation Accuracy: {final_acc:.2f}%\nFinal Meta Accuracy: {final_meta_acc:.2f}%\nMacro Precision: {macro_p:.4f}\nMacro Recall: {macro_r:.4f}\nMacro F1: {macro_f:.4f}\n")

    print("\n" + "="*40)
    print(f"All outputs saved to: {output_dir.absolute()}")
    print("="*40)

if __name__ == '__main__':
    main()