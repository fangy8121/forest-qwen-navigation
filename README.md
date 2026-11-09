# Forest Qwen Navigation

林下无人机全覆盖巡检的仿真基线（Isaac Sim）。程序在线识别树行、行末和通道，并列出能走的下一步；Qwen3.5-4B 根据中文指令在图上选一个；局部规划负责避障和跟踪。

本仓库是 **2026-11-09 基线**，只包含运行、评测和测试需要的代码，不含地图、模型权重和运行记录。

## 方法

1. **指令解析**：Qwen 把一句中文指令解析成固定字段：任务类型、顺序（S 形或任意）、通道范围、做完是否恢复原任务。
2. **在线建图**：正装 MID-360 雷达的点云拟合树行，得到通道、行末（已确认或未确认）和覆盖账本。账本只累计无人机实际飞过的区间。
3. **候选**：程序看无人机在哪，列出能走的走法（开局观察、靠近入口、沿通道巡检、补巡缺口、换行、返航等 13 种），删掉不能走、没有新增、不允许和不必要的，再加一个悬停。程序不按任务顺序删选项。
4. **Qwen 点选**：输入一张全局图和一段文字，输出 `{"next_aisle":3,"choice":"A"}`。同一决策按不同的候选顺序问 2–3 遍，把概率相加。只在飞机停稳时询问。
5. **任务层**：选定后沿通道中心线每 4 m 取一个目标点，飞到尽头；换行只绕已确认的行末。
6. **局部规划**：30 s 滚动点云、ESDF（0.2 m 网格）、A*（0.4 m 网格）、B 样条，Lee 位置控制器跟踪。约每 0.1 s 预检一次前方轨迹和刹停路径，不通过就刹停。

## 结果（仿真）

| 地图组 | 张数 | 通过 | 地图目录 |
|---|---|---|---|
| 测试 | 7 | 3 | train/020、021、022、023、025、026、027 |
| 验证 | 4 | 4 | test/000、train/008、val/000、val/002 |
| 训练（较早版本） | 9 | 6 | train/005、006、007、010、012、013、015、016、017 |

地图目录名沿用生成时的划分，评测按实际用途重新分组。两张需要借邻行绕行的地图（train/009、train/014）不在评测范围内。

测试集 7 张都没有碰撞，4 张巡完了全部通道，Qwen 选错通道 1 次。4 张失败都和局部规划、恢复有关。

解读时请注意：
- 定位和离地高度用的是仿真真值，结果偏乐观。
- 样本很少，3/7 的 95% 区间约为 16%–75%。
- 只测过人工林（树行近似平行，行距 2.5–7 m）。
- 评测的是路线覆盖，不代表树木表面被看清。

## 环境

测试环境：Ubuntu 22.04，RTX 4070 Ti SUPER 16 GB。物理仿真用 CPU PhysX，显存留给 Qwen。

- **仿真：** Isaac Sim 5.1.0 和 Isaac Lab 0.54.2 的 conda 环境，Python 3.11、PyTorch 2.7.0+cu128、TorchRL 0.10.0、TensorDict 0.10.0、Hydra 1.3.6。
- **Qwen：** 在仿真环境之上建一个继承系统包的虚拟环境，只额外安装 `cfg/forest_qwen35_requirements.txt`。

```bash
conda activate env_isaaclab        # 已装好 Isaac Sim 5.1 和 Isaac Lab 的环境
pip install -e .
python -m venv --system-site-packages .venvs/forest_qwen35
.venvs/forest_qwen35/bin/pip install -r cfg/forest_qwen35_requirements.txt
.venvs/forest_qwen35/bin/python scripts/download_forest_models.py --sizes 4B   # 下载到 models/Qwen3.5-4B
```

## 地图

地图约 780 MB，没有放进仓库。用同一批种子重新生成，默认输出到 `cfg/forest_coverage.yaml` 里的 `outputs/01_森林巡检仿真跑通/maps`：

```bash
scripts/generate_baseline_maps.sh
```

- 每张地图的目录、种子和障碍组合，已用 `--dry-run` 和原地图清单逐项核对一致。
- 本基线没有重新生成一遍、逐文件核对几何。
- 生成器只发布通过检查的地图，可能比上表多出几张，评测只用上表的 20 张。
- 树木模板参数在 `catkin_ws/src/survey_simulator/models/out_test/scene*.csv`，生成器和两项测试会读取它。

## 运行

飞一张图：

```bash
.venvs/forest_qwen35/bin/python scripts/fly_forest_coverage.py task.scene.map_id=train/020 max_seconds=3000.
```

逐张飞多张图（不要并行）。结果写到 `outputs/baseline/runs`，日志在 `outputs/baseline/logs`：

```bash
scripts/run_maps.sh test train/020 train/021 train/022 train/023 train/025 train/026 train/027
```

- **结果文件：** 日志最后一行 `COVERAGE_RESULT` 是本次结论。每次运行的目录里有 `result.json`、`decisions.jsonl`、`task_events.jsonl`，每次决策的输入图和回答在 `language_decisions/decision_*/`。
- **独立复核：** `python scripts/report_forest_native_coverage.py <运行目录>`。
- **其他指令：** 只巡部分通道、任意顺序、去指定通道、中途返回再继续，用对应配置，例如 `--config-name forest_coverage_subset`。这些场景只在较早版本飞过（前两种各 1 次，都通过）。

## 测试

```bash
.venvs/forest_qwen35/bin/python -m pytest tests
```

CPU 上约 1 分钟，不需要 Isaac Sim 窗口。

## 代码位置

| 内容 | 位置 |
|---|---|
| 主配置 | `cfg/forest_coverage.yaml`（继承链见各文件的 `defaults`） |
| 飞行入口 | `scripts/fly_forest_coverage.py` → `omni_drones/navigation/forest/coverage_flight.py` |
| 树行、通道、行末 | `omni_drones/navigation/forest/online_structure.py` |
| 覆盖账本 | `omni_drones/navigation/forest/coverage.py` |
| 候选生成 | `coverage_candidates.py`，以及 `coverage_mission.py` 中的 `_plan_native` |
| 提示词和多顺序合并 | `coverage_prompt.py` |
| 任务层 | `coverage_mission.py`、`task_supervisor.py` |
| Qwen 进程 | `scripts/forest_subgoal_model_worker.py`、`qwen_subgoal.py` |
| 局部规划和预检 | `local.py`（A*）、`observed_esdf.py`、`stable_spline.py`、`spline_runtime.py`、`lower_flight.py` |
| 仿真环境 | `omni_drones/envs/single/forest_survey.py`、`cfg/task/ForestSurvey.yaml` |
| 地图生成 | `forest_generator/`、`scripts/generate_forest_multirow_dataset.py` |
| 决策题库 | `scripts/build_forest_decision_bench.py`、`scripts/eval_forest_decision_bench.py`（题库数据不在仓库） |

## 已知问题

- 通道中段受阻时，候选只有“继续往前”和“悬停”，没有绕行或换行。
- 正在穿过目标点时，如果前方 1–4 m 的目标都被挡住，任务会直接结束，不会刹停后重新决策。
- 有两张测试图在没有树枝的地方路线突然爬高，原因还没查清。
- Qwen 偏爱第一个字母，所以要换顺序问多遍；绕一点的说法（如否定句）会解析错。

## 来源与许可

- 代码基于 [OmniDrones](https://github.com/btx0424/OmniDrones)（MIT，见 `LICENSE`）。
- 树木网格、纹理和树木参数 CSV 来自 [survey_simulator](https://github.com/tlin442/survey_simulator)（GPL-3.0），来源记录见 `forest_generator/assets/provenance.json`。
