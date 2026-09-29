# 内部评分参与程序选择：393 / 394

固定协议见 docs/GUIDED_PROTOCOL.md。候选程序与隐藏位置均在提取/拟合/验证世界上决定，之后才生成最终测试世界。

## 验收

- 全套 54 项测试通过。
- 两个正式实验独立重放通过，包括候选重新合成、读出重新拟合、验证分数重算和最终五环境指标复核。
- 方法是有限候选重排，不是无限程序空间的全局搜索。

## 实际选择

| Seed | 候选数 | 获得正内部支持的候选 | 内部选择是否改变程序 |
|---|---:|---:|---|
| 393 | 3 | 2 | False |
| 394 | 2 | 2 | False |

## 候选的验证证据

| Seed | 候选 ID | 复杂度 | 验证保真 | 内部支持 | 受支持位置 | 行为/MDL 分数 | 加入内部支持后的分数 |
|---|---|---:|---:|---:|---|---:|---:|
| 393 | b3fdc90d2189 | 39 | 0.766 | 0.002641 | representation | 0.726625 | 0.726757 |
| 393 | c9b5d88b6ca2 | 35 | 0.767 | 0.000000 | 无 | 0.731602 | 0.731602 |
| 393 | a20097af3b77 | 17 | 0.745 | 0.036455 | head_tanh | 0.728117 | 0.729940 |
| 394 | 8f8327565268 | 33 | 0.775 | 0.009797 | representation | 0.742391 | 0.742880 |
| 394 | bacefb3abfc4 | 19 | 0.777 | 0.024750 | representation | 0.758344 | 0.759581 |

## 最终独立测试

| Seed | 环境 | 方法 | 网络准确率 | 程序准确率 | 程序保真 |
|---|---|---|---:|---:|---:|
| 393 | test_id | with_internal | 74.0% | 65.6% | 76.4% |
| 393 | test_id | without_internal | 74.0% | 65.6% | 76.4% |
| 393 | test_function | with_internal | 75.4% | 63.1% | 70.1% |
| 393 | test_function | without_internal | 75.4% | 63.1% | 70.1% |
| 393 | test_noise | with_internal | 77.1% | 62.3% | 72.9% |
| 393 | test_noise | without_internal | 77.1% | 62.3% | 72.9% |
| 393 | test_scale | with_internal | 50.0% | 69.1% | 49.4% |
| 393 | test_scale | without_internal | 50.0% | 69.1% | 49.4% |
| 393 | test_intervention | with_internal | 60.4% | 64.1% | 80.3% |
| 393 | test_intervention | without_internal | 60.4% | 64.1% | 80.3% |
| 394 | test_id | with_internal | 68.2% | 63.5% | 77.9% |
| 394 | test_id | without_internal | 68.2% | 63.5% | 77.9% |
| 394 | test_function | with_internal | 72.1% | 57.4% | 66.2% |
| 394 | test_function | without_internal | 72.1% | 57.4% | 66.2% |
| 394 | test_noise | with_internal | 64.1% | 58.0% | 61.7% |
| 394 | test_noise | without_internal | 64.1% | 58.0% | 61.7% |
| 394 | test_scale | with_internal | 48.0% | 67.2% | 45.7% |
| 394 | test_scale | without_internal | 48.0% | 67.2% | 45.7% |
| 394 | test_intervention | with_internal | 67.4% | 66.0% | 79.1% |
| 394 | test_intervention | without_internal | 67.4% | 66.0% | 79.1% |

## 解释边界

- 内部和无内部方法共享候选、验证数据及复杂度项；内部方法额外使用映射计算，不宣称总计算预算相同。
- 纠正后的内部支持必须超过随机和打乱目标控制；证据不足时支持为零，不宣称选出了受支持的隐藏位置。
- 选择相同、支持为零或最终测试退化均保留，不据测试结果调大内部权重或删去失败候选。
- 即使选到了不同程序，也不能据此证明原始统计运算、完整内部算法或普遍因果原则已恢复。
- 图结构、机制父集/算子恢复，以及原始数据到统计算子的完整反编译仍须继续实现和检验。
