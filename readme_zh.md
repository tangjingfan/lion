# Language-instructed Object Navigation（中文版）

基于 LION-Bench 改造的 Landmark-RxR rollout + 数据清洗流水线。
Habitat-Lab 的 `rgbds_agent` 提供 `rgb`、`depth`、`semantic` 三路观测。

本 README 端到端覆盖四个核心环节：

1. **Rollout** — 让 agent 跑完所有选中的 episode。
2. **Filter pipeline 0-3** — 把 `(instruction_id, sub_idx)` 收敛到值得后续 grounding 的那些。
3. **Target instance selection** — 给每个幸存 sub-path 选出一个 MP40 instance id 作为导航目标。
4. **Filter pipeline 4 + rescue** — 基于最终 target 做语义粒度过滤，并可选用 YOLO-World（VLM 兜底）救回部分粗粒度实例。

> 英文原版见 [readme.md](readme.md)，两份内容保持同步。

## Setup

配置文件分两类：

- **`configs/rollout/rollout_landmark_rxr.yaml`** — 机器级别路径（Landmark-RxR JSON、MP3D 场景、connectivity、输出根目录）。每台机器配置一次即可。
- **`configs/selection/<exp>.yaml`** — 单次实验的配置（`expname`、episode 列表、agent、viz 开关）。每个实验一个文件，下面所有命令都基于一个具体的 selection。

最终产出落在 `results/{split}_{expname}/`。README 通篇用的示例 selection：

```bash
SEL=configs/selection/one_scene_partial_val_unseen.yaml
# split: val_unseen, expname: partial_one_scene
# scan : X7HyMhZNoso, ~19 instructions
# → results/val_unseen_partial_one_scene/
```

`results/{run}/` 下，**除 `filters/` 外** 每个工具的子目录都按 scan 进一步切分，这样多场景实验各自的产物互不污染：

```
results/val_unseen_partial_one_scene/
  rollout_viz/X7HyMhZNoso/...
  rewrite/X7HyMhZNoso/...
  scene_categories/X7HyMhZNoso/...
  partition/X7HyMhZNoso/...
  target_instances/X7HyMhZNoso/...
  filters/                 # 跨 scan，不分子目录
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

## 2. Filter pipeline（stage 0-3）

过滤流水线先把 `(instruction_id, sub_idx)` 收敛到值得 grounding 的那些。这里先跑 stage 0-3；stage 4 需要等第 3 章 target instance selection 完成后再跑。每一步都写到 `results/{run}/filters/`：

```
NN_{name}.yaml          # 通过的 (ep, sub) 列表，selection 格式
NN_{name}_dropped.yaml  # 被丢掉的 + 原因
audit.json              # 每条 (ep, sub) 在各阶段的状态
current.yaml            # 指向最新阶段的 keep 文件（symlink）
```

`current.yaml` 是下游所有工具通过 `--from_yaml` 读的入口。
这些 stage 都是**纯 filter**——每个 stage 接收一个幸存集合，输出一个更小的幸存集合。词表准备和 visibility 标注放在第 3 章。

```bash
CURRENT=results/val_unseen_partial_one_scene/filters/current.yaml
```

注意：`filters/current.yaml` / `03_partition.yaml` 这类 survivor 文件要传给 `--from_yaml`，不要传给 `--config`。`--config` 只用于 `configs/rollout/rollout_landmark_rxr.yaml` 这种 rollout 配置。

完整执行顺序是：

```text
filter stage 0-3
        │
        ▼
target instance selection（第 3 章）
        │
        ▼
可选：VLM semantic rescue（第 4 章）
        │
        ▼
filter stage 4 semantic_granularity（第 4 章）
```

#### 2.0 Snapshot（不丢任何东西）

```bash
bash scripts/00_record_original.sh --from_yaml "$SEL"
```

写入 `filters/00_snapshot.yaml` 并把 `current.yaml` 指过去。

#### 2.1 跨楼层过滤

丢掉参考路径竖直跨度 > 0.5 m 的 episode（视为跨楼层）。

```bash
bash scripts/01_filter_multi_floor.sh --from_yaml "$SEL"
```

#### 2.2 LLM 重写

对每条幸存 sub-instruction，让 LLM 产出 `landmark + landmark_category + landmark_instruction + spatial_instruction`，外加 per-component 分解。然后丢掉那些"很难指代/落到 MP3D 上"的 landmark（黑名单包括 door、window 等通用词）。

```bash
GEMINI_API_KEY=... bash scripts/02_rewrite_subinstruction.sh --from_yaml "$SEL"
bash scripts/03_blacklist_landmark.sh --from_yaml "$SEL"
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
bash scripts/04_partition.sh --from_yaml "$SEL"
```

产物：

```
partition/X7HyMhZNoso/{instruction_id}/partition.json
partition/X7HyMhZNoso/{instruction_id}/{instruction_id}.png
```

stage 3 跑完后，`current.yaml` 指向 `03_partition.yaml` —— 这就是 target instance selection 的输入集合，第 3 章用的就是它。

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
bash scripts/05_get_object_list.sh --from_yaml "$SEL" --objects_only
```

产物：

```
scene_categories/X7HyMhZNoso/objects.json
```

### 3b. 重映射 landmark mapping（LLM，按 scan）

让 LLM 重新做一次映射，候选词汇**只能**从这个 scan 的 `objects.json` 里挑。结果**原地覆盖** `landmark_mapping_filtered.json`。

```bash
GEMINI_API_KEY=... bash scripts/06_refine_landmark_mapping.sh --from_yaml "$SEL"
```

> **关于 `max_tokens` 为什么调得这么大**：`configs/rewrite/rewrite_subinstructions.yaml` 里现在用的是 `gemini-2.5-flash`，这是 thinking 模型，内部推理也会占用 `max_tokens` 预算。这一步的回复又是一个覆盖整个 scan 所有 mention 的大 JSON，所以 `max_tokens` 设到了 `32768`，给 thinking 和输出都留足空间。预算太小（比如 4096）会把 JSON 截断成不闭合的字符串，报 `Unterminated string ... could not parse JSON`。不带 thinking 的 `gemini-2.0-flash` 已经对 Gemini API 新账号下线了。

### 3c. 在 partition 点枚举候选 target instance

对每条幸存 `(ep, sub)`，在 **partition 点**（这一段 sub-path 跟下一段之间的转折节点，通常是 `partition.json` 里的 `virt:...` 虚拟节点）渲染一张 360° 语义全景，列出当中匹配 landmark 类别的所有可见 MP40 instance。`uniqueness` 根据**该视角下可见 instance 的数量**决定，不是看全场景总数——只有"agent 站在转折点能不能一眼看清楚"这个量决定了下游是否还需要做消歧。默认还会为每个候选在同一个 pose 渲染一张 mask PNG。

```bash
bash scripts/07_list_potential_instances.sh --from_yaml "$SEL"
# 不存可视化图（更快）：
bash scripts/07_list_potential_instances.sh --from_yaml "$SEL" --no_save_viz
# 调高每个 instance 的最小像素阈值：
bash scripts/07_list_potential_instances.sh --from_yaml "$SEL" --min_pixel_count 100
```

读取：
- `filters/current.yaml`
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
bash scripts/08_get_potential_instance.sh --from_yaml "$SEL"
# 打印所有多候选 sub-path：选了谁、各候选距离多少
bash scripts/08_get_potential_instance.sh --from_yaml "$SEL" --print_multi
# 用更轻的 .house 顶视图代替 Habitat 渲染（debug 用）
bash scripts/08_get_potential_instance.sh --from_yaml "$SEL" --viz_mode topdown
# 不存可视化图
bash scripts/08_get_potential_instance.sh --from_yaml "$SEL" --no_save_viz
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

## 4. Filter pipeline（rescue + stage 4）

### 4a. Pixel-grounded semantic rescue（可选，推荐先跑）

把只落到 MPCAT40 粗粒度（`appliances` / `lighting` / `objects` / …）的目标用开放词表检测救回细类别（`stove` / `refrigerator` / `lamp` / …）。主路径是 YOLO-World，VLM 仅在 detector 找不到时作为兜底（默认关）。

输入 3c 的 `target_instances/{scan}/target_instances.json` + rollout `frames.jsonl`。流程：筛出只落到粗粒度 semantic label 的 sub-path → 在同 pose 重渲染 RGB + raw semantic panorama → 用 landmark 短语（自动扩同义词，如 `fridge → {fridge, refrigerator}`、`stove → {stove, oven, cooktop, range, ...}`）作 prompt 跑 YOLO-World → 对每个 detection 按 score 降序在 semantic buffer 里做类别感知的 instance 恢复（优先选 bbox 内 MPCat40 名字能反向包含 detector 细类的 instance，例如检测到 `stove` 时优先 bbox 内的 `appliances` instance），命中第一个就停。可以救 `target_instance_ids: []` 的样本，因为 instance id 是从检测框反查的。

```bash
# Dry run：先看会送哪些 coarse sub-path 进 detector
bash scripts/09_vlm_rescue.sh \
    --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml \
    --dry_run

# 实际跑（首次会自动下载 ~340MB YOLO-World 权重 + CLIP）：
bash scripts/09_vlm_rescue.sh \
    --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml

# 可选 VLM 兜底（YOLO 没检出时才调 Gemini）：
GEMINI_API_KEY=... bash scripts/09_vlm_rescue.sh \
    --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml \
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

回路：4b stage 4 启动时自动读取 rescue 字典，命中 `(episode_id, sub_idx)` 或 `target_instance_id` 即标 `ok_rescued` 进 survivor YAML —— 不需要额外接线。

旧的 mask-based rescue 仍可用：

```bash
bash scripts/build_semantic_rescue_categories.sh --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml --dry_run
```

它只处理已有 `target_instance_ids` 的样本，让 VLM 在已有 mask 上起更细类别；适合审计已有 selection，但救不了没 instance id 的样本。

### 4b. 语义粒度过滤（最终 drop，不需要 LLM）

这一步处理的是 semantic taxonomy 的粒度问题，而不是可见像素问题。例如 instruction 里说 `stove`，但 MP3D/MPCAT40 的语义标注只把它归到上位类 `appliances`，没有单独的 `stove` semantic label。这样的目标即使可见，也不能证明它被语义系统单独标出来。

这个 stage 会先查 `target_instances/{scan}/semantic_rescue_categories.json`：如果 VLM rescue 已经把这个 `(episode_id, sub_idx)` 或选中的 instance id 救成更细类别，就保留并在 audit 里记为 `ok_rescued`；否则才 drop。

规则：

- 读取最终选中的 `target_instance_ids`，看这个 instance 的 semantic label。
- 如果它只属于配置的粗粒度 label（默认 `appliances`、`objects`、`furniture`、`lighting`），并且没有被 rescue 成更细类别，就 drop。
- 如果选中 instance 有更具体的 semantic label，或 VLM rescue 成功，则保留。

```bash
# rescue 之后，用 stage 3 survivor set 作为输入做最终 semantic filter：
bash scripts/10_filter_semantic_granularity.sh --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml

# 只看统计，不写 04_semantic_granularity.yaml / 不改 current.yaml：
python src/check/filter_semantic_granularity.py \
    --config configs/rollout/rollout_landmark_rxr.yaml \
    --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml \
    --report_only
```

输出：

- `filters/04_semantic_granularity.yaml` —— 通过语义粒度检查和 rescue 检查的 survivor set。
- `filters/04_semantic_granularity_dropped.yaml` —— 仍然被丢掉的样本，原因是 `coarse_semantic_label`，并记录 `landmark`、`semantic_labels`、`coarse_label`、`target_instance_ids`。
- `filters/current.yaml` —— 指向 `04_semantic_granularity.yaml`。
