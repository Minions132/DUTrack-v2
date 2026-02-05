# DUTrack GOT-10k 测试运行指南

## 📁 项目结构说明

### 只保留的性能指标
- **AUC (Success Curve)**: 衡量跟踪框与真实框的IoU重叠
- **P-R Curve (Precision-Recall)**: 基于置信度分数的精度-召回曲线

### 配置文件
```
experiments/dutrack/
├── dutrack_256_got.yaml           # GOT-10k 原始配置
├── dutrack_256_got_baseline.yaml  # Baseline (FLOW_WINDOW_SIZE=1)
├── dutrack_256_got_ours.yaml      # Ours (FLOW_WINDOW_SIZE=5)
├── dutrack_256_full.yaml          # LaSOT/TNL2K 完整配置
├── dutrack_256_baseline.yaml      # LaSOT Baseline
└── dutrack_256_ours.yaml          # LaSOT Ours
```

### 分析脚本
```
tracking/
├── analyze_got10k.py              # GOT-10k 专用分析 (推荐使用)
├── compare_baseline_ours.py       # 通用对比脚本
└── analysis_results.py            # 原始分析脚本
```

---

## 🚀 快速开始

### 前提条件
1. 确保GOT-10k测试集在 `data/got10k/test/` 目录下
2. 确保模型权重文件存在

### 步骤1: 运行Baseline测试
```bash
cd /home/m1n1ons/projects/dev/DUTrack

# 使用 GOT-10k Baseline 配置运行测试
python tracking/test.py dutrack dutrack_256_got_baseline --dataset got10k_test
```

### 步骤2: 运行Ours测试
```bash
# 使用 GOT-10k Ours 配置运行测试 (改进版)
python tracking/test.py dutrack dutrack_256_got_ours --dataset got10k_test
```

### 步骤3: 生成对比曲线
```bash
# 单独分析某个配置
python tracking/analyze_got10k.py --config dutrack_256_got

# 对比 Baseline vs Ours
python tracking/analyze_got10k.py --compare \
    --baseline_config dutrack_256_got_baseline \
    --ours_config dutrack_256_got_ours
```

---

## 📊 输出说明

### 测试结果保存位置
```
output/test/tracking_results/
└── got10k_test/
    └── dutrack/
        └── dutrack_256_got_baseline/
            ├── GOT-10k_Test_000001.txt        # 跟踪结果
            └── GOT-10k_Test_000001_all_scores.txt  # 置信度分数 (P-R曲线需要)
```

### 图表保存位置
```
output/got10k_plots/
├── got10k_success_comparison.png  # AUC/Success 曲线对比
└── got10k_pr_comparison.png       # P-R 曲线对比
```

---

## 🔧 配置差异说明

| 参数 | Baseline | Ours | 说明 |
|------|----------|------|------|
| FLOW_WINDOW_SIZE | 1 | 5 | 时序窗口大小 |
| FLOW_UPDATE_INTERVAL | 10 | 10 | 更新间隔 |
| SAVE_SCORES | true | true | 保存置信度(P-R需要) |

**核心差异**: 
- Baseline: 单帧更新 (原始论文方法)
- Ours: 多帧时序共识更新 (改进方法)

---

## ⚠️ 注意事项

1. **GOT-10k测试集特殊性**: GOT-10k测试集没有真实标注，需要提交到官方评测服务器
   - 如果只是本地测试流程，可以使用 `got10k_val` 验证集
   
2. **使用验证集测试**:
   ```bash
   # 如果没有测试集的GT，使用验证集
   python tracking/test.py dutrack dutrack_256_got_baseline --dataset got10k_val
   ```

3. **P-R曲线需要置信度文件**: 确保配置中 `SAVE_SCORES: true`

4. **首次运行会很慢**: 需要加载模型和处理所有序列

---

## 📈 预期结果

根据论文，在GOT-10k测试集上的预期性能：
- **DUTrack-256**: AO = 76.7%
- **DUTrack-384**: AO = 77.8%

改进方法（Temporal Flow Consensus）预期在某些场景下提升1-2%
