# UCI Wine Dataset - Reduct Testing

## 数据集信息

UCI Wine Dataset 包含178个样本，13个化学成分属性和1个类别标签（3种葡萄酒类别）。

### 属性列表：
1. alcohol
2. malic_acid
3. ash
4. alcalinity_of_ash
5. magnesium
6. total_phenols
7. flavanoids
8. nonflavanoid_phenols
9. proanthocyanins
10. color_intensity
11. hue
12. od280/od315_of_diluted_wines
13. proline
14. class (决策属性)

## 测试流程

### 1. 数据预处理

脚本：`wine_reduct_simple.py`

步骤：
1. 从不同类别随机选取样本
2. 对所有数值属性进行 Min-Max 归一化到 [0, 1]
3. 将归一化值四舍五入到 0.25 精度（即 0.0, 0.25, 0.5, 0.75, 1.0）
4. 移除 class 列，只保留数值属性用于约简测试

### 2. 约简测试

使用 `fuzzy_fca_reduct.py` 中的函数：

- `FuzzyFormalContext` - 创建模糊形式上下文（用于FCA）
- `FuzzyRoughContext` - 创建模糊粗糙上下文（用于RST）
- `is_fca_reduct()` - 判断子集是否是FCA约简
- `is_rst_reduct()` - 判断子集是否是RST约简

### 3. 测试方法

**测试单个属性移除**：
- 逐个移除属性，测试剩余属性是否能保持原上下文的约简性质
- 对于每个属性集 Y'：
  - `X'` = 所有对象
  - `Y'` = 所有属性 - {被移除属性}

**测试组合属性移除**：
- 测试同时移除2个属性的情况
- 由于计算复杂度，限制测试的组合数量

## 预期结果

### 使用 Gödel 剩余格

Gödel 格不满足双重否定律（¬¬0.5 = 1 ≠ 0.5），因此：
- FCA 和 RST 的约简结果可能不同
- 可能找到 **divergence（分歧）** 情况

### 使用 Łukasiewicz 剩余格

Łukasiewicz 格满足双重否定律（¬¬a = a 对所有 a∈[0,1]），因此：
- FCA 和 RST 的约简结果应该一致
- 不应出现分歧

## 运行测试

```bash
# 快速测试
python wine_reduct_simple.py

# 数据探索
python explore_uci_wine.py
```

## 理论背景

根据论文 Theorem 4.7：

- **当剩余格满足双重否定律时**：
  - FCA 约简和 RST 约简通过否定相互等价
  - 两种理论给出一致的约简结果

- **当剩余格不满足双重否定律时**（如 Gödel 格）：
  - FCA 约简和 RST 约简可能不同
  - 两种理论可能给出矛盾的结论

## 模糊值精度

测试中使用的模糊值精度：
- **细粒度**: 0.0, 0.25, 0.5, 0.75, 1.0

归一化公式：
```
normalized = (x - min) / (max - min)
rounded_to_025 = round(normalized * 4) / 4
```

## 输出示例

```
================================================================================
UCI WINE DATASET - REDUCT TEST (FAST)
================================================================================

Loaded wine dataset: (178, 14)
Classes: [1 2 3]
Class distribution:
1    59
2    71
3    48

Selected 2 samples from classes: [1, 2]

Testing attribute removal:
--------------------------------------------------------------------------------
Attribute                 | FCA Reduct | RST Reduct | Divergence
--------------------------------------------------------------------------------
alcohol                   |     NO     |    YES     |       YES
malic_acid               |     NO     |     NO     |        NO
...

SUMMARY
================================================================================
FCA reducts found: 2
RST reducts found: 5
Divergences (FCA != RST): 3

*** DIVERGENCE FOUND! ***
This demonstrates that with Godel lattice (which fails
double negation), FCA and RST can give different reduct results.
```

