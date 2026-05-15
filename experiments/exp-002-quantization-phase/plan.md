# Exp-002: 量化临界 bit-width Pilot 实验

## 目的

验证"存在临界 bit-width b*，低于它 SAE 特征不再可恢复"的假说。
在 Pythia-160M 上做快速 pilot，为服务器上的完整实验（Gemma-2 2B/9B）提供预备数据。

## 假设

1. 随着 bit-width 降低（FP32→INT8→INT4→INT3→INT2），SAE 特征恢复率单调下降
2. 下降可能在某个 bit-width 附近出现加速（如果存在相变）
3. 稀有特征（低 firing rate）可能比常见特征更早崩溃（Michaud 预测）

## 方法/配置

### 模型
| 参数 | 值 |
|------|-----|
| 模型 | EleutherAI/pythia-160m-deduped |
| 层 | Layer 6 MLP output |
| d_model | 768 |

### SAE
| 参数 | 值 |
|------|-----|
| 来源 | exp-001 Run 1, seed 0 |
| 架构 | TopK |
| d_sae | 4096 |
| K | 32 |

### 量化
| 参数 | 值 |
|------|-----|
| 量化方法 | HQQ (Half-Quadratic Quantization) |
| Bit-widths | FP32 (baseline), INT8, INT4, INT3, INT2 |
| Group size | 64 |
| 校准数据 | 无（HQQ 不需要） |

### 数据
| 参数 | 值 |
|------|-----|
| 推理 tokens | 500,000（足够统计但省时间） |
| 序列长度 | 1024 |
| 数据集 | EleutherAI/the_pile_deduplicated (streaming) |

### 度量
1. **特征激活相关性**: 量化前后匹配特征的 Pearson r
2. **NMSE 变化**: 量化激活经 SAE 重建的误差
3. **特征恢复率**: 量化后还能被 SAE 检测到的特征比例
4. **按 firing rate 分层分析**: 高/中/低频特征分别的恢复率

### 计算资源
| 参数 | 值 |
|------|-----|
| 设备 | MPS (M4 Max 48GB) |
| 峰值内存 | ~8GB |
| 预计时间 | ~30 分钟 |
