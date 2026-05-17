# Language-instructed Object Navigation（中文版）

基于 LION-Bench 改造的 Landmark-RxR rollout + 数据清洗流水线。
Habitat-Lab 的 `rgbds_agent` 提供 `rgb`、`depth`、`semantic` 三路观测。

本 README 端到端覆盖四个核心环节：

1. **Rollout** — 让 agent 跑完所有选中的 episode。
2. **Filter pipeline 00-04** — 把 `(instruction_id, sub_idx)` 收敛到值得后续 grounding 的那些。
3. **Target instance selection** — 给每个幸存 sub-path 选出一个 MP40 instance id 作为导航目标。
4. **VLM pixel-grounded rescue** — 用 YOLO-World（VLM 兜底）把 MPCAT40 粗粒度（`appliances` / `lighting` / …）的目标救回细类别（`stove` / `lamp` / …）。
5. **Blacklist 拯救 + 整合 dataset** — 给 blacklist 剔的 sub-path（landmark 太泛如 `wall`/`door`/`room`）从 sub-path 终点视野里挑一个新 landmark，然后把每条 sub-trajectory（原始 + 合成）拼成 `dataset.json`。

> 英文原版见 [readme.md](readme.md)，两份内容保持同步。

## Setup

配置文件分两类：

- **`configs/rollout/rollout_landmark_rxr.yaml`** — 机器级别路径（Landmark-RxR JSON、MP3D 场景、connectivity、输出根目录）。每台机器配置一次即可。
- **`configs/selection/<split>/<exp>.yaml`** — 单次实验的配置（`expname`、episode 列表、agent、viz 开关）。每个实验一个文件，按 `split` 组织（如 `val_unseen/`）；下面所有命令都基于一个具体的 selection。

最终产出落在 `results/{split}_{expname}/`。README 通篇用的示例 selection：

```bash
SEL=configs/selection/val_unseen/one_scene_partial.yaml
# split: val_unseen, expname: partial_one_scene
# scan : X7HyMhZNoso, ~19 instructions
# → results/val_unseen_partial_one_scene/
```

`results/{run}/` 下，pipeline 的 canonical state 是 run 根目录下的一份 `survivor.yaml`；每个工具的子目录按 scan 进一步切分，多场景实验各自的产物互不污染；`filters/`（drop 诊断 + audit）跨 scan 共享：

```
results/val_unseen_partial_one_scene/
  survivor.yaml            # canonical 当前 state（每个 filter stage
                           # 原地覆盖这个文件）
  rollout_viz/X7HyMhZNoso/...
  rewrite/X7HyMhZNoso/...
  scene_categories/X7HyMhZNoso/...
  partition/X7HyMhZNoso/...
  target_instances/X7HyMhZNoso/...
  filters/                 # per-stage NN_*_dropped.yaml + audit.json
```

## 1. Rollout

让 agent 把选中的 episode 全部跑一遍，并记录每一步的观测。


```bash
bash scripts/rollout.sh \
    --config    configs/rollout/rollout_landmark_rxr.yaml \
    --selection "$SEL"
```

产出在 `results/val_unseen_partial_one_scene/rollout_viz/X7HyMhZNoso/`：

```
{instruction_id}/{sub_idx:03d}/{step:04d}.png    # 每帧 RGB+depth+semantic
frames.jsonl                                      # 每帧元数据
results.json                                      # 每个 episode 的 metric + 汇总
replay.yaml                                       # 实际跑过的 instruction_ids
config_used.yaml                                  # 合并后的实际生效配置
```

注意：`sub_idx` **从 0 开始**，文件夹名是 `000/`、`001/`，不是 `001/`、`002/`。

`frames.jsonl` 会被 filter 阶段的 partition 步骤读取（步骤 2.3），让分界点反映实际 rollout 的运动几何，而不是参考路径。

## 2. Filter pipeline（步骤 00-04）

过滤流水线先把 `(instruction_id, sub_idx)` 收敛到值得 grounding 的那些。这里先跑 00-04；最终 filter（步骤 10）等第 3 章 target instance selection 完成后再跑。

每个 stage 都把 canonical 状态写到 run 根目录下的 `survivor.yaml`，诊断信息留在 `filters/`：

```
{run}/survivor.yaml                  # 通过的 (ep, sub) 列表，selection 格式
                                     # 下游工具通过 --exp 自动 merge 这个
{run}/filters/NN_{name}_dropped.yaml # 被丢掉的 + 原因（debug 用）
{run}/filters/audit.json             # 每条 (ep, sub) 跨 stage 的状态
```

这些 stage 都是**纯 filter**——每个 stage 接收当前幸存集合，输出更小的幸存集合。词表准备和 visibility 标注放在第 3 章。

`--exp` 既接 selection YAML 路径（如 `configs/selection/val_unseen/one_scene_partial.yaml`），也接裸 `expname`（如 `one_scene_partial`）。两种写法下游都会自动 merge `survivor.yaml`，用户不需要手动传 survivor 文件。

完整执行顺序：

```text
filter 步骤 00-04（record / multi_floor / rewrite / blacklist / partition）
        │
        ▼
target instance selection（第 3 章，脚本 05-08）
        │
        ▼
VLM pixel-grounded rescue（第 4 章，脚本 09）
        │
        ▼
把 rescue 结果合回 target_instances.json（第 4 章，脚本 10）
        │
        ▼
blacklist 拯救：换个 landmark（第 5 章，脚本 13）
        │
        ▼
整合（原始 + 合成）→ dataset.json（第 5 章，脚本 11）
        │
        ▼
漏斗报告（第 5 章，脚本 12 —— 任意时刻可重跑）
```

#### 2.0 Snapshot（不丢任何东西）

```bash
bash scripts/00_record_original.sh --exp "$SEL"
```

在 `survivor.yaml` 不存在时把种子 selection 写进去（如果已有更窄的 in-progress 状态则不覆盖）。同时写 `filters/00_original_dropped.yaml`（空）+ `filters/audit.json`。

#### 2.1 跨楼层过滤

丢掉参考路径竖直跨度 > 0.5 m 的 episode（视为跨楼层）。

```bash
bash scripts/01_filter_multi_floor.sh --exp "$SEL"
```

#### 2.2 LLM 重写

对每条幸存 sub-instruction，让 LLM 产出 `landmark + landmark_category + landmark_instruction + spatial_instruction`，外加 per-component 分解。然后丢掉那些"很难指代/落到 MP3D 上"的 landmark（黑名单包括 door、window 等通用词）。

```bash
GEMINI_API_KEY=... bash scripts/02_rewrite_subinstruction.sh --exp "$SEL"
bash scripts/03_blacklist_landmark.sh --exp "$SEL"
```

产物：

```
rewrite/X7HyMhZNoso/{instruction_id}/sub_instructions_rewritten_filtered.json
rewrite/X7HyMhZNoso/landmark_mapping_filtered.json    # mention → [labels]
```

这里的 `landmark_mapping_filtered.json` 是 rewriter 自己 per-component 猜的，可能把别的场景词汇也拉进来。第 3 章的 3b 会把它清洗一遍再进入 visibility 标注。

#### 2.3 Partition（用 rollout frames）

把每条幸存 sub-path 切成 spatial 段 + landmark 段，分界点从 `rollout_viz/{scan}/frames.jsonl` 推出来（几何说明见下）。partition 出错的 sub-path 在这一步被丢掉。

```bash
bash scripts/04_partition.sh --exp "$SEL"
```

产物：

```
partition/X7HyMhZNoso/{instruction_id}/partition.json
partition/X7HyMhZNoso/{instruction_id}/{instruction_id}.png
```

步骤 04 跑完后，`survivor.yaml` 里就是 sub-path 级别的幸存集合 —— 这就是 target instance selection 的输入。

##### Partition 几何

Partition 优先读 `rollout_viz/{scan}/frames.jsonl`。它会累计 rollout 的有向转角（`TURN_RIGHT` 为正、`TURN_LEFT` 为负）：

- 如果在前进距离达到阈值前 `|累计转角| >= turn_thresh_deg`，认为该 sub-path "带转向"，分界点放在跨过转角阈值的那一步**之后再前进 0.3 m**；
- 否则视为纯直行，分界点放在 rollout sub-path 起点**前进 0.3 m**处；
- 如果完全没有 rollout frames，回退用参考路径走同样的距离阈值。

## 3. Target instance selection

给每条幸存 sub-path 挑一个 MP40 instance id 作为导航目标。前两步先给 rewriter 的 mention→label 映射准备一份"只来自本 scan"的干净词表，然后标注 visibility，选出最终 target。跑完 3d 后，继续第 4 章做语义粒度过滤。

```text
list_scene_categories      ← 缓存 scan 物体词表
        │
        ▼
refine_landmark_mapping    ← LLM 重写 landmark_mapping_filtered.json
        │
        ▼
list_target_instances      ← 枚举候选 instance + uniqueness 标签
        │
        ▼
select_target_instances    ← 选 sub-path 终点距离最近的那个
                              （uniqueness == unique 时直接 skip）
```

### 3a. 缓存场景的物体词表

解析每个 scan 的 MP3D `.house` 文件，拿到实际出现的 MPCAT40 类别列表，和 rollout viz 使用的标签一致。这份词表是 3b refine 之后 `landmark_mapping_filtered.json` **唯一允许的**词汇来源。

```bash
bash scripts/05_get_object_list.sh --exp "$SEL" --objects_only
```

产物：

```
scene_categories/X7HyMhZNoso/objects.json
```

### 3b. 重映射 landmark mapping（LLM，按 scan）

让 LLM 重新做一次映射，候选词汇**只能**从这个 scan 的 `objects.json` 里挑。结果**原地覆盖** `landmark_mapping_filtered.json`。

```bash
GEMINI_API_KEY=... bash scripts/06_refine_landmark_mapping.sh --exp "$SEL"
```

> **关于 `max_tokens` 为什么调得这么大**：`configs/rewrite/rewrite_subinstructions.yaml` 里现在用的是 `gemini-2.5-flash`，这是 thinking 模型，内部推理也会占用 `max_tokens` 预算。这一步的回复又是一个覆盖整个 scan 所有 mention 的大 JSON，所以 `max_tokens` 设到了 `32768`，给 thinking 和输出都留足空间。预算太小（比如 4096）会把 JSON 截断成不闭合的字符串，报 `Unterminated string ... could not parse JSON`。不带 thinking 的 `gemini-2.0-flash` 已经对 Gemini API 新账号下线了。

### 3c. 在 partition 点枚举候选 target instance

对每条幸存 `(ep, sub)`，在 **partition 点**（这一段 sub-path 跟下一段之间的转折节点，通常是 `partition.json` 里的 `virt:...` 虚拟节点）渲染一张 360° 语义全景，列出当中匹配 landmark 类别的所有可见 MP40 instance。`uniqueness` 根据**该视角下可见 instance 的数量**决定，不是看全场景总数——只有"agent 站在转折点能不能一眼看清楚"这个量决定了下游是否还需要做消歧。默认还会为每个候选在同一个 pose 渲染一张 mask PNG。

```bash
bash scripts/07_list_potential_instances.sh --exp "$SEL"
# 不存可视化图（更快）：
bash scripts/07_list_potential_instances.sh --exp "$SEL" --no_save_viz
# 调高每个 instance 的最小像素阈值：
bash scripts/07_list_potential_instances.sh --exp "$SEL" --min_pixel_count 100
```

读取：
- `survivor.yaml`（通过 `--exp` 自动 merge）
- `rewrite/X7HyMhZNoso/{instruction_id}/sub_instructions_rewritten_filtered.json`
- `rewrite/X7HyMhZNoso/landmark_mapping_filtered.json`（推荐 3b refine 过的版本）
- `partition/X7HyMhZNoso/{ep}/partition.json`

写入：
- `target_instances/X7HyMhZNoso/target_instances.json` —— 每条 (ep, sub) 一份：
  - `landmark`、`semantic_labels`、`matched_category`、`matched_categories`、`matched_by`、`pixel_count`、`pixel_fraction`
  - `candidates[]` —— 每个 `{id, category, n_pixels}`（开 viz 时还带 `viz_path` / `viz_visible_pixels`）
  - `uniqueness` ∈ {`unique`, `ambiguous`, `not_visible`, `no_match`, `partition_pos_unresolvable`}
- `target_instances/viz/X7HyMhZNoso/{ep}/sub_{NNN}_cand_{IID}.png` —— 每个可见候选一张：在 partition 点渲染的 RGB + 语义全景，下方拼了一条 target mask 高亮该候选。

### 3d. 选出 target instance

这一步的渲染位置在 **sub-path 终点** (`sub_path_nodes[-1]`，最后一个 step)，**不是** partition 点。原因：agent 最终应该停在 target 附近，所以候选打分用"到终点的距离"。

规则：

- **只有 1 个可见 instance** → 直接选它（`view_unique`）。
- **多个可见 instance** → 选 AABB center 离 sub-path 终点最近的那个（`view_nearest`）。每个 scan 只会加载 Habitat 场景一次，从语义标注里读 instance 的 habitat 坐标中心。
- **多个可见但拿不到位置** → 回退到像素最多的那个（`view_nearest_fallback`）。

```bash
bash scripts/08_get_potential_instance.sh --exp "$SEL"
# 打印所有多候选 sub-path：选了谁、各候选距离多少
bash scripts/08_get_potential_instance.sh --exp "$SEL" --print_multi
# 用更轻的 .house 顶视图代替 Habitat 渲染（debug 用）
bash scripts/08_get_potential_instance.sh --exp "$SEL" --viz_mode topdown
# 不存可视化图
bash scripts/08_get_potential_instance.sh --exp "$SEL" --no_save_viz
```

写入：
- `target_instances/X7HyMhZNoso/target_instances.json` —— 每条 (ep, sub) 一份：
  - `target_instance_ids` —— 选中的 instance id（数组，目前只放一个）
  - `status` —— 上面三个判定之一
  - `selection_distance` —— 选中 instance 到 sub-path 终点的距离（米）
  - `candidate_distances` —— 每个候选 id → 距离的字典
  - `candidates[]` —— 从 visibility 透传过来的完整候选列表
- `target_instances/viz_last_frame/X7HyMhZNoso/{ep}/sub_{NNN}_last.png` —— rollout 最后一帧的原始可视化。
- `target_instances/viz_partition/X7HyMhZNoso/{ep}/sub_{NNN}_id_{IID}.png` —— partition 点的候选 target mask 可视化。
- `target_instances/viz_last_frame_instance/X7HyMhZNoso/{ep}/sub_{NNN}_id_{IID}.png` —— 如果 Habitat 渲染可用，在 sub-path 终点渲染 RGB + 语义全景图，并高亮选中的 instance。

## 4. VLM pixel-grounded rescue（流水线最后一步）

把只落到 MPCAT40 粗粒度（`appliances` / `lighting` / `objects` / …）的目标用开放词表检测救回细类别（`stove` / `refrigerator` / `lamp` / …）。主路径是 YOLO-World，VLM 仅在 detector 找不到时作为兜底（默认关）。

输入 3c 的 `target_instances/{scan}/target_instances.json` + rollout `frames.jsonl`。流程：筛出只落到粗粒度 semantic label 的 sub-path → 在同 pose 重渲染 RGB + raw semantic panorama → 用 landmark 短语（自动扩同义词，如 `fridge → {fridge, refrigerator}`、`stove → {stove, oven, cooktop, range, ...}`）作 prompt 跑 YOLO-World → 对每个 detection 按 score 降序在 semantic buffer 里做类别感知的 instance 恢复（优先选 bbox 内 MPCat40 名字能反向包含 detector 细类的 instance，例如检测到 `stove` 时优先 bbox 内的 `appliances` instance），命中第一个就停。可以救 `target_instance_ids: []` 的样本，因为 instance id 是从检测框反查的。

记录到输出里的 `category` 字段是**原 instruction 里的 landmark 短语**（也就是当初喂给 YOLO 的 prompt），不是 detector 自己的 class name。所以 `stove` 始终是 `stove`，即便 YOLO-World 实际命中的是 `cooktop` class。

```bash
# Dry run：先看会送哪些 coarse sub-path 进 detector
bash scripts/09_vlm_rescue.sh \
    --exp "$SEL" \
    --dry_run

# 实际跑（首次会自动下载 ~340MB YOLO-World 权重 + CLIP）：
bash scripts/09_vlm_rescue.sh \
    --exp "$SEL"

# 可选 VLM 兜底（YOLO 没检出时才调 Gemini）：
GEMINI_API_KEY=... bash scripts/09_vlm_rescue.sh \
    --exp "$SEL" \
    --enable_vlm_fallback
```

依赖：`pip install ultralytics`（拉 ultralytics + CLIP，自动在 `lion` env 跑 CUDA）。

主要 CLI：

- `--yolo_model`：默认 `yolov8l-worldv2.pt`；可换 `yolov8x-worldv2.pt`（更准、更慢）或 `s` / `m` 版（更快）。
- `--yolo_conf`：默认 `0.10`。
- `--yolo_imgsz`：默认 `1024`，匹配全景图宽。
- `--yolo_device`：覆盖 torch device，缺省走 ultralytics 自动。
- `--enable_vlm_fallback`：开关；开了才需要 `--api_key` / `GEMINI_API_KEY`。
- `--sample_radius` / `--search_radius`：instance 恢复时 tight / wide 两层 shell 的搜索范围。

输出：

- `target_instances/{scan}/semantic_rescue_categories.json` —— 每 scan 一份。核心 lookup 是 `instances["{instance_id}"] -> {category, confidence, is_rescuable, semantic_category, grounding_method, landmarks, examples, image_paths}`。`grounding_method` 标 `yolo_world` 或 `vlm_fallback`。
- `target_instances/{scan}/vlm_pixel_grounding/{episode_id}/sub_{NNN}_{rgb,bbox,point,mask}.png` —— 干净 RGB、detector bbox、bbox 中心、恢复出来的 MP3D instance mask 叠图。
- `target_instances/{scan}/vlm_pixel_grounding/vlm_pixel_grounding_summary.{json,png}` —— grounding 结果总览 JSON + contact sheet（每行 4 张缩略图，最后一列就是 mask 叠图；detector category 与 sem label 不匹配时标 `[MISMATCH]` 红字）。

旧的 mask-based rescue 仍可用：

```bash
bash scripts/build_semantic_rescue_categories.sh --exp "$SEL" --dry_run
```

它只处理已有 `target_instance_ids` 的样本，让 VLM 在已有 mask 上起更细类别；适合审计已有 selection，但救不了没 instance id 的样本。

### 把 rescue 结果合回 target_instances.json

rescue 步骤产出的是 side-car（`semantic_rescue_categories.json`）。Step 10 把里面的命中信息合并回 `target_instances/{scan}/target_instances.json`，让 canonical 的 target selection 记录也反映 rescue 的结果：

- 之前是 `not_visible`（`target_instance_ids` 空）且 rescue 救到了 → 填入 `instance_id`、`status = "rescued"`、记下 `rescue_landmark` / `rescue_category`（= 当初 YOLO prompt 用的 landmark 短语）。
- 之前已经有 target → 不改 target，只追加 `rescue_landmark` / `rescue_category` / `rescue_instance_id`，让下游知道 rescue 也确认过这个。

```bash
bash scripts/10_apply_rescue.sh --exp "$SEL"
```

幂等：重跑会先清掉旧的 `rescue_*` 注释再写新的，跑几次结果都一样。原先 4b 的「粗粒度语义过滤」已经移除：rescue 没救出来的样本不会再被剔除——下游消费者自己看 `target_instances/{scan}/semantic_rescue_categories.json` 决定怎么用就好。

## 5. 整合存活的 sub-trajectory

读 `survivor.yaml` + 之前各 stage 写的 artifact，把每条存活的 sub-trajectory 拼成一条记录：text（完整 instruction + sub-split + landmark / spatial 分解）、几何（`sub_path_nodes` / `spatial_path` / `landmark_path` / 起始 heading / partition kind）、选中的 target instance + landmark 在 partition 点是否可见、rescue 相关注释、可视化文件路径。纯聚合——不调 LLM / 仿真器 / detector。

```bash
bash scripts/11_consolidate.sh --exp "$SEL"
```

输出：

- `results/{run}/dataset.json` —— 顶层是 JSON list，每条对应一个存活的 `(scan, instruction_id, sub_idx)`。每条记录是 rewrite / partition / target_instances 三个 JSON 的字段并集，外加从 Landmark-RxR 数据集直接拿到的 instruction 原文，再加 `landmark_source ∈ {"original", "synthesized"}` 字段供下游过滤。

### Blacklist 拯救（合成新 landmark）

跟 09 的 rescue 平行，但目标是 **blacklist 剔（03）** 掉的 sub-path —— 那些原 landmark 太泛（"wall"/"door"/"room"/"doorway"）。这里不是重新 ground 原 landmark，而是**直接换一个**。候选要求：

  1. sub-path 终点视野里可见（≥ 200 像素）
  2. **逐渐靠近**（终点距离 比 partition 点近至少 0.5 m）
  3. MPCAT40 类别 concrete（不在 `wall`/`door`/`window`/`floor`/`ceiling`/... 黑名单里）
  4. 优先取该视角下**唯一可见的**那一类（"the X" 无歧义）

```bash
bash scripts/13_rescue_blacklist.sh --exp "$SEL"
bash scripts/11_consolidate.sh --exp "$SEL"   # 救完重跑 consolidate
```

输出：

- `target_instances/{scan}/blacklist_rescue.json` —— 每 scan 一份，记录新 landmark、合成的 sub-instruction（目前用简单模板 `"<spatial>. Walk to a <landmark>."`）、新 target instance id，外加 approach / uniqueness 统计。

consolidate 这一步把这些作为额外 record 加进 `dataset.json`，标 `landmark_source = "synthesized"`，并在 `synthesized_from` 字段里记下原 landmark + drop 原因。
