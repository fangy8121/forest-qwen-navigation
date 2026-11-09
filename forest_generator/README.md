# 森林地图生成器

## 当前底层验证与诊断地图

在线导航底层的[成果与运行指南](../docs/forest_lower.md)已更新：5 张冻结开发图上的 40 项任务全部通过，另完成 80 m 双阻挡长行、原有森林 65 m 长行和故障停止验证。它们使用公开坐标任务，尚未接 Qwen，也不等于整图巡检覆盖。

80 m 长行诊断地图单独生成，保留平地与圆柱的诊断属性；已有目录无需重建，生成器拒绝覆盖：

```bash
.venvs/forest_qwen35/bin/python scripts/build_forest_lower_long_map.py \
  --output outputs/forest_vln/executable_lower/long_maps
```

导出 `long/000/` 下的 `scene.usd`、`scene_mesh.npz`、`map.json` 和 `public_mission.json`。该图用于检查长航程、滚动地图和两处在线绕障；树枝、起伏地形的验证另在原生森林中进行。运行与原始证据见[底层实验入口](../outputs/00_旧路线证据/forest_vln/executable_lower/README.md)。

### 短任务地图的生成入口

`python -m forest_generator generate-executor-pilot` 导出到
`outputs/forest_vln/executor_pilot/maps/`，拒绝覆盖已有目录。
先看该目录的 `README.md`：`diagnostic/000` 是四种短运动任务的固定圆柱诊断地图，
`native/<seed>` 是复用原生树模板、枝干和起伏地形的 12×12 米森林预览。
初始原生预览的几何导出记录不等于飞行验收；后续 40 项底层任务的冻结输入与物理结果见上方实验入口。
每图有 `overview.png`、`map.json`、`scene.usd`、`scene_mesh.npz`、`tasks.json`，
原生地图另有 `terrain.npz`。`manifest.json` 保存种子、配置来源与文件哈希。
当前推荐使用 `scripts/fly_forest_lower.py` 运行连续任务，默认结果写入 `outputs/forest_vln/executable_lower/continuous/runs/`。早期 `scripts/fly_forest_subgoal.py` 入口保留用于历史对照。

目前有**两条任务生成流程，共用原来的森林、树枝和地形生成器**。下一阶段多行巡检优先使用新版；原有单通道流程继续用于局部能力测试和历史对照。

| 流程 | 支持内容 | 入口 |
| --- | --- | --- |
| 多行巡检（当前推荐） | 5 种整图障碍组合，包含多条树行、行末换行和完整教师路线 | `python -m forest_generator generate-multirow` |
| 原有单通道任务 | 4 类局部任务：`lateral`、`vertical`、`choice`、`mixed` | `python -m forest_generator generate` |

这不是 9 种互不相关的森林。新版是在原生森林上组合局部任务；原来的底层生成器仍在使用。新版 5 种组合**不能**传给原 CLI 的 `--tasks`。

## 多行巡检地图

默认生成**不等长树行**：名义行长 65–75 m，起点在 4 m 范围内错位，相邻行长度有弱相关性；默认场地约 119 × 36 m。沿用目前地图的原生树模板、树干尺寸、株距、行距、枝干数量/长度/高度/角度分布，以及起伏地形参数；不删树枝、不压平地面来提高可飞性。这仍是行列人工林模型，不声称等同于实测天然林。

每行保存 `start_x`、`end_x`、`length_m`。先采样名义边界，再在边界内按原株距分布放树，不截断已生成的树或枝干。通道出口依据相邻树行的实际树干和枝干包络确定，避免短行结束后就穿过旁边的长行。转弯采用一条位置、切向和二阶导连续的五次曲线，保持原速度、加速度、jerk 和偏航限制。整条参考轨迹仍需通过完整三维网格 0.75 m 中心净空、离地高度和任务门检查。

旧批次 30 张短行地图已于 2026-10-19 清理。`--layout legacy` 可在新目录生成旧式布局（60 × 36 m 场地、40 × 32 m 原生林分）。新旧布局都保留原生枝干和起伏地形，名义离地高度约 2.5 m。

| 组合 | 障碍安排 | 当前批次每图障碍事件数 |
| --- | --- | ---: |
| `light` | 一行左右可选绕行，一行侧向绕行 | 4 |
| `balanced` | 可选绕行、两行侧向绕行、一行连续障碍 | 9 |
| `sequences` | 两行连续障碍、一行可选绕行 | 8 |
| `mixed_bypass` | 两行可选绕行、一行连续障碍、一行侧向绕行 | 9 |
| `neighbor_detour` | 一处借邻行绕行，加可选绕行和侧向绕行 | 5 |

每图至少保留两条未额外添加任务障碍的原生通道；这些通道仍有原生树枝和地形。障碍事件数量不是树干数量。新版当前没有专门加入旧版 `vertical` 的上下横木任务，但保留正常升降和地形跟随。

### 从哪里查看

当前预览入口：五张紧凑边界地图。默认平滑 U 弯；`task.scene.route_mode=stop` 切换到停稳、转向、横移、再停稳对齐的对照模式。两种模式共用此前五张不等长地图的树木、树枝和地形，未扩充完整合集。`turn_modes.png` 展示两种路线的区别。

紧凑版按完整树枝网格外缘至少 2 m、两种航线外缘至少 3 m 的并集确定地面边界，再向外对齐原地形网格。实际裁去外围地面三角面，保留区域的原三角面、树木和两种时参数化参考轨迹均与源文件核对。不会为了缩小地图删枝、平地或移动障碍。3 m 是这批已验证参考路线的场地余量，不是后续任意在线任务的安全承诺。

几何复核见 `verification.json`。原 双模式五图 的完整飞行结果属于裁剪前版本，不冒充紧凑版的完整飞行验证。不等长初版五图和 30 张旧布局地图已清理；现存地图中的历史来源路径及原始哈希仍作为生成记录保留。

```text
outputs/02_紧凑边界五图/
  index.html                 # 地图总览入口
  train/000 … train/004       # 五种组合各一张，固定开发种子
  manifest.json              # 布局配置、种子、源码哈希和发布状态
  verification.json          # 重新读取导出文件后的独立复核
  index.json                 # 仿真自动选图入口
```

每张地图优先查看 `overview.png`（俯视图与完整教师路线）和 `terrain_route.png`（地形及路线高度）。仿真与后续采集使用 `scene.usd`、`scene_mesh.npz`、`terrain.npz`、`reference_trajectory.npz`、`map.json` 和纹理文件。

`map_stop.json`、`reference_stop.npz` 是同一森林的停稳对照参考。停稳模式会依据实际位置误差 < 0.10 m、速度 < 0.10 m/s、航向误差 < 5°、角速度 < 0.12 rad/s 连续满足 0.3 s 后切换阶段；等待超过 20 s 明确终止。平滑 U 弯在换行处不强制停车。这是手动选择的两种预览模式，目前没有自动判断空间后切换模式的任务规划器。

根目录 `manifest.json` 是正式地图清单，`verification.json` 是复核结果，`candidate_reviews.jsonl` 保留被拒绝候选的参数和原因。被拒绝候选不计入正式地图数，不能当作已经完成的训练样本。

### 打开 Isaac Sim 观看飞行

对已经发布为 PASS 的训练地图 `train/000`，观看无人机沿完整教师参考路线飞行：

```bash
cd /home/fang/OmniDrones
conda activate env_isaaclab
python scripts/collect_forest.py \
  headless=false batch=false \
  'task.scene.map_dir="outputs/02_紧凑边界五图"' \
  task.scene.map_id=train/000 \
  task.scene.split=train \
  task.scene.route_mode=continuous \
  task.flight.height=2.5 \
  task.env.max_episode_length=90000 \
  max_seconds=900 \
  output=outputs/multirow_flights
```

这是**教师参考路线的动力学飞行与 LiDAR 采集，不是 Qwen 策略飞行**。`headless=false` 打开仿真窗口；`max_seconds=900` 是最大仿真时长，路线完成或触发终止条件会提前结束。地图的 `validation.duration` 给出参考时长，还需加上起飞稳定时间；默认 300 秒不足以看完长行地图。

换地图只需修改 `task.scene.map_id`，并将 `task.scene.split` 改成相应集合。新版开发批次使用 `train/000`～`train/004`，没有验证或测试地图。飞行记录保存在 `outputs/multirow_flights/` 下的独立航次目录；查看 `result.json` 中的结束原因，`reference_complete` 才表示完成整条参考路线。新预览使用轨迹切线航向与偏航速度/加速度前馈，仍由 Lee 控制器、旋翼和 PhysX 推进，不在飞行中直接设置位姿。

暂停时参考时钟和采集一起冻结，点击 Stop 会保存记录并结束本次航次。点云显示限量、降频，扫描压缩使用有界后台队列；原始扫描不降采样或丢帧。`result.json.timing_seconds` 分别记录扫描、排队、显示、物理和渲染耗时，不保证所有机器都能实时运行。

### 阶段标注与换行数据

`map.json.reference_phases` 完整、连续地划分 `INSPECT`（行内）、`EXIT`（出行）、`TURN`（转弯）、`ENTRY`（入行），包括阶段起止时间、世界坐标端点、通道编号、长度和平均速度。出行/入行段默认各取通道末端/首端 3 m 的过渡距离。新预览以每帧保存的 `reference_time` 定位阶段，不能简单用仿真时间减去固定起飞时长代替，因为实际停等可延长执行时间。`execution.json` 保存参考时钟、执行状态与停稳门限通过记录。

全程保留原始 LiDAR。新预览采集帧包含 `phase`、`execution_state` 与 `default_navigation_training`；只有行内且非停等的帧可进入默认训练样本筛选。`build_forest_dataset.py` 会将其他帧标为 `mission_transition` 并排除监督样本，同时保留其因果历史观测。允许筛选仍不等于通过教师标签审核；本轮不启动 Qwen 训练。

用已有五张森林重新生成双模式预览（输出目录须不存在）：

```bash
python scripts/build_forest_stop_turn_maps.py --output outputs/dual_mode_preview
```

从已有双模式五图生成紧凑边界预览（同一批种子，只裁地面）：

```bash
python scripts/compact_forest_preview_maps.py --output outputs/compact_preview
```

同图提速对照可运行：

```bash
python scripts/benchmark_forest_headlands.py \
  --source outputs/02_紧凑边界五图/train/000 \
  --output outputs/headland_comparison
```

该命令使用同一森林、同一通道参考和相同运动上限，仅重新连接换行；原地图不变。输出 `comparison.json`、`comparison.png` 和独立参考轨迹；结果属于离线教师轨迹，不是闭环实飞成绩。

历史旧 `train/000` 同图对照：五次换行合计 **159.82 → 42.44 秒（减少 73.4%）**，完整路线 **468.22 → 341.95 秒**；两条曲线的完整网格净空均通过。报告 · 对照图。该报告源地图已清理，上述命令现在使用当前紧凑地图，不会复现这些历史数值。新版因行长增加，总巡检路程也增加，不能直接用其总时长与旧短行地图比较。

### 生成新地图、续跑与复核

以下命令同样在项目根目录、激活 `env_isaaclab` 后执行。已有地图无需重新生成。默认参数对应原来的 20/5/5 张地图和种子：训练 `20261700–20261719`、验证 `20261720–20261724`、测试 `20261725–20261729`。原脚本入口仍可使用，与正式入口共享参数和生成逻辑。

生成新一批独立森林，选择全部五种组合（默认），指定数量和未使用过的起始种子：

```bash
python -m forest_generator generate-multirow \
  --counts 20 5 5 --seed-start 20261800 \
  --output outputs/multirow_20261800 --max-attempts 6
```

只生成指定组合，例如 5 张借邻行绕行的训练地图：

```bash
python -m forest_generator generate-multirow \
  --profiles neighbor_detour --seeds 20261900 20261901 20261902 20261903 20261904 \
  --split train --output outputs/multirow_detour_20261900 --max-attempts 6
```

在任一命令末尾加 `--dry-run` 可预览种子、组合和目录分配，不写文件、不生成地图。每个种子只生成一张地图，所选组合在每个集合内按顺序循环分配，**不会把同一森林复制成五张独立训练地图**。数量不是组合数的整数倍时，各组合数量最多相差一张。

| 参数 | 含义 |
| --- | --- |
| `--output` | 输出根目录，默认 `outputs/generated_multirow` |
| `--layout` | `variable`（默认，不等长行和连续换行）或 `legacy`（旧布局） |
| `--row-length MIN MAX` | 名义行长范围，默认 `65 75` m，要求 30 ≤ MIN ≤ MAX |
| `--row-start-jitter` | 起点错位的半范围，默认 2 m，即总范围 4 m |
| `--profiles` | 选择一种或多种组合，默认全部五种 |
| `--counts TRAIN VAL TEST` | 各集合地图数，默认 `20 5 5`；允许某个集合为 0 |
| `--seed-start` | 连续种子的起点，默认 `20261700`，依次分给训练、验证、测试 |
| `--seeds`、`--split` | 指定种子列表及其集合，默认集合为 train；不可与 counts/seed-start 混用 |
| `--dry-run` | 只打印分配清单，不生成或写文件 |
| `--workers` | 并行生成进程数，默认 3 |
| `--max-attempts` | 本次每图最多尝试的障碍布局候选数，默认 3，不保证全部通过 |
| `--only train/000 val/000` | 只处理本批清单中的指定地图 |
| `--retry-failed` | 对尚未发布的失败地图继续尝试新布局，保留已有正式地图 |

同一版本续跑会跳过已完成地图；种子、组合、数量、布局配置或生成源码与已有清单不一致时会停止，避免覆盖和混合版本。新版五图开发批次的续跑参数是 `--counts 5 0 0 --max-attempts 6`；默认 `20 5 5` 应使用其他新目录。新批次使用未使用的种子；不同输出目录之间不会自动检查种子重复，需统一管理种子分配。重试只调整附加障碍布局，不更换原生森林种子，也不删除原生树枝来强行通过。**仅更换输出目录会重现同一批森林，不会产生新的独立数据集。**

重新检查已保存的几何、完整参考轨迹和运动约束，并生成高度图：

```bash
python scripts/audit_forest_multirow_dataset.py --plots
```

可选 `--repair-timing` 会对轻微超出运动限制的参考轨迹统一延长时间，保持几何路线不变，并备份原文件；普通复核无需此参数。

离线验收包括原生树木和行边界可重现性、附加树干与原生树干/树枝间隔、任务绕行、连续轨迹完整网格净空（要求 0.75 m）、运动约束及阶段划分。**PASS 只表示离线地图验收通过**，不代表真实扫描采集、滚动六点教师验证、Qwen 训练或闭环飞行通过。后续还需验证观测覆盖和制动空间。

完整教师路线可使用地图几何；它不等于学生输入。不得把地图真值边界、未来参考路线或未来地形高度直接送给 Qwen。在线 LiDAR 地形估计与多行任务管理仍需单独完成对接。

新版主要代码：[`multirow_tasks.py`](multirow_tasks.py) 安排障碍，[`build_forest_multirow_native.py`](../scripts/build_forest_multirow_native.py) 构建地图与完整参考路线，[`generate_forest_multirow_dataset.py`](../scripts/generate_forest_multirow_dataset.py) 管理批量生成，`audit_forest_multirow_dataset.py` 复核保存结果。

## 原有单通道任务

以下命令和参数仅适用于原有流程。它适合单独验证左右绕障、上下绕障、起点偏移等能力，也保留旧实验的复现入口；不承担新版完整多行巡检数据集的生成。

生成一张 **hard 难度的混合任务训练地图**。在 Bash 终端执行：

```bash
cd /home/fang/OmniDrones
conda activate env_isaaclab
python -m forest_generator generate --seeds 0 --tasks mixed --difficulty hard --split train
```

这条命令使用种子 `0`，生成 36 m 短地图，保存到 `outputs/forest/maps/hard/mixed/000000/`。`--split train` 将地图归入训练集，**不会筛掉任何指定种子**。

**生成多个地图时，只需修改种子：** 将 `--seeds 0` 改为 `--seeds {0..30}`，即种子 0～30，共 31 张 `mixed` 地图；`{0..50}` 则是 51 张。范围包含两端，`--seeds 0 50` 表示只生成编号 0 和 50 两个种子。如果已经生成种子 0，继续生成时用 `{1..30}`，避免重复。

**需要全部四类任务时，去掉 `--tasks mixed`。** 每个种子生成 4 张地图，因此 0～30 对应 31 个森林、124 张任务地图；0～50 对应 51 个森林、204 张任务地图。所有地图仍为 hard 难度，全部归入训练集。

后续生成验证集或测试集时，使用未用于训练的新种子，将 `--split train` 改成 `--split val` 或 `--split test` 即可。无需现在预留或生成其他集合，已有地图的归属也不会随新增种子改变。

以上数量以几何验证全部通过为前提；失败会明确报告，不会悄悄换种子补齐。成功地图见 `outputs/forest/maps/index.json`，失败原因见命令打印的报告文件。已有同名地图会报错，不会覆盖。**地图生成完成后，回到[主 README 第 2 步“采集数据”](../README.md#2-采集数据)继续。**

### 单通道地图内容

| 类别 | 内容 | 每个种子的地图数 |
| --- | --- | ---: |
| `lateral` | 左右交替绕行，种子决定先左还是先右 | 1 |
| `vertical` | 上绕与下穿交替，种子决定先上还是先下 | 1 |
| `choice` | 两处均可左右绕行的障碍，由规划器选择通道 | 1 |
| `mixed` | 左右绕行与上绕、下穿组合 | 1 |

选择全部四类时，每个种子共尝试生成 4 张地图；指定单类任务时每个种子只生成 1 张。普通直行路段也包含在地图中。任务由障碍几何决定，Qwen 不接收隐藏的任务名或参考路径。

`vertical` 每张地图都包含上绕和下穿。短布局 hard 地图中，偶数种子为“上绕—下穿—上绕”，奇数种子为“下穿—上绕—下穿”。起点类型不改变同一种子的障碍排列，不再自动生成 `_reverse` 对照。指定种子 0～29、每个种子一种起点时，共尝试生成 30 张上下地图。

<details>
<summary>参数、地形和验证说明</summary>

### 可调整的参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--seeds` | 实际生成的种子编号；Bash 的 `{0..30}` 包含两端 | `0` |
| `--split` | 将全部指定种子归入 `train`、`val` 或 `test` | `train` |
| `--tasks` | 从四类任务中选择 | 全部四类 |
| `--difficulty` | `easy`、`medium`、`hard` | `easy` |
| `--variant` | `center` 常规起点；`offset` 起点及朝向偏移；`approach` 障碍前 4 m 起步 | `center` |
| `--layout` | `short` 短布局；`long` 长布局 | `short` |
| `--output` | 地图根目录 | `outputs/forest/maps` |

同一种子、同一配置的任务、难度和起点变体共享 `scene_id`，必须归入同一集合。在同一地图根目录再次生成该森林的变体时，指定冲突的集合会报错。划分记录在 `map.json` 和 `index.json` 中，采集后的 `scene_truth.json` 与数据集转换沿用此记录；转换时发现同一 `scene_id` 跨集合也会报错。仅缺少划分记录的旧数据使用原有固定哈希规则。

论文实验中应分别报告独立森林数、任务地图数和有效样本数：同一种子的 4 张任务地图共享森林，不能当作 4 个独立森林。保留配置、种子、代码版本、生成报告和地图清单以便复现。更改配置或布局会改变 `scene_id`；跨配置实验仍应保持训练、验证、测试种子互不重叠，不能只依靠 ID 检查排除场景相关性。

### 地形与障碍

生成配置在 `config.json`，短布局由 `layout_config` 调整。短地图长 36 m，常规起点的参考路线通常约 30～34 m，起点变体可能更短；长布局沿用配置，默认长 100 m。名义离地高度至少为 2.5 m。

短布局采用两层 Perlin 地形：小尺度振幅 1.2 m、低频振幅上限 4 m、低频频率 0.08、坡度范围 ±0.06。树根和参考路线按真实三角地面计算高度。

短布局中，`easy/medium/hard` 的障碍组数分别为 2 / 2～3 / 3；`choice` 固定两组，`mixed` 固定三组。难度还改变水平障碍的直径和偏移，不保证实际飞行难度严格递增。

每处上下障碍是一根直径 0.5 m 的水平横木，连接通道边的承托树干。下绕横木中心比当地名义飞行高度高 0.65 m，上绕则低 0.65 m；保留少量平面偏转，两端等高。USD 显示、碰撞和 LiDAR 使用同一份网格。

任务区四周补充 6 m 背景树林，地面四周留 8 m 边缘。外围树木采用独立随机流，保持任务区原有树行与同种子森林的对应关系。完整生效参数保存在每张地图的 `map.json`。

此前两个种子的检查批次中，任务区域地形高差约 2.8～3.2 m，下穿参考路线比名义高度低约 0.44～0.59 m，地面上距最近树干超过 5 m 的面积约占 1%～3%。这些是历史测量，新种子需要重新检查。

### 验证与输出

每张地图检查三维 ESDF 连通性、通道内实际开口、连续平滑轨迹和完整网格净空，参考路线的几何净空要求大于 0.75 m。上下类只验证当前种子指定的高度排列，不要求另一套相反排列也可行。

上下障碍处不强制上绕或下穿；两个方向都安全时保留选择。参考路线在障碍之间回到通道中心附近，有上下障碍时还回到名义高度附近。模型路径按实际几何评估，不按场景名称判错。

`map.json` 的 `validation` 保存净空、路线长度、持续时间、障碍过点高度与方向（`height_crossings`），以及轨迹采样点距任务障碍 X 坐标不超过 4 m 的比例（`task_zone_fraction`）。后者衡量任务区域覆盖，不等同于有效训练样本比例。

每张地图包含 `scene.usd`、`scene_mesh.npz`、`terrain.npz`、`reference_trajectory.npz`、`map.json` 和树木/树行 CSV。只有通过验证的地图进入根目录的 `index.json`，本次生成报告位于相应难度目录。

几何验证失败会记录原因并继续其他地图，最后返回非零状态；成功地图保留。遇到已存在的地图目录会立即停止，不自动覆盖、跳过或续跑。离线验证通过后仍需要参考飞行和 LiDAR 采集。

历史检查记录保留在 `outputs/forest/checks/`。此前的地图和参考航次已按要求清理，历史结果不能替代新地图的验证。

</details>

模板和纹理来源见 [provenance.json](assets/provenance.json)。采集、标签生成与训练见[操作指南](../docs/forest_survey.md)。

当前完整巡检流程先在紧凑地图 `004` 进行单图试采集，保留换行与绕障审核素材。
输入、标签隔离及复现命令见[单图试采集说明](../docs/forest_mission_pilot.md)；该审核格式尚未作为正式训练集使用。
# 执行器冻结开发集

子目标执行器的下一批开发数据可通过以下命令生成；它使用新目录，拒绝覆盖已有实验：

```bash
python -m forest_generator generate-executor-suite --output outputs/forest_vln/new_dev40/maps --source-maps outputs/forest_vln/executor_pilot/maps
```

该命令导出 24 个圆柱诊断任务和 16 个原生森林短段，共 5 张图。原生网格、枝干和地形保持不变；先按固定候选顺序检查短段几何可行性，再冻结任务。每张图保存 `tasks.json`、`episodes.json`、`tasks.png` 和仅用于离线审计的可行路径，整批保存文件哈希。几何通过不等于闭环飞行通过。

运行与结果入口见[40 项开发测试](../outputs/00_旧路线证据/forest_vln/executor_dev40/README.md)。
