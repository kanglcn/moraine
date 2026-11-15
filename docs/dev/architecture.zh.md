---
title: 架构
---

!!! info "这是仓库里的 `ARCHITECTURE.md`（英文）"
    代码的地图：分层、每个模块的职责、依赖方向。`tests/test_architecture.py` 保证每个模块都在表里、分层规则成立。
    面向用户的整体介绍见[概念](../start/concepts.md)：从磁盘上的数据，到在 CPU 或 GPU 上按块处理它们的 worker，
    再到命令行和 pipeline。

--8<-- "ARCHITECTURE.md"
