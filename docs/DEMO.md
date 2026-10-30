# Demo: Trick-or-Treat Sorter（15:30，每队 5 分钟）

台词用英文（引号内照念），操作说明用中文。评分权重：Functionality 40 · Ambition 20 · Creativity 20 · Modularity 15 · Halloween 5。
代码 14:30 冻结。下面的「真机」都有 session 里的 job 记录或照片；没上过真机的标「离线」。

---

## 1. 讲稿（5:00）

### 0:00–0:30 开场：一句话架构（Ambition / Creativity）

> "Type any list of things — 'red lollipop, yellow block, tape' — and the arm finds each one on a messy table and drops it in the pumpkin bucket.
> Viam runs everything on the robot: one module, `sorter-team:sorter:grasp-by-description`, drives the xArm, the gripper and the RealSense.
> MongoDB Atlas is the robot's memory: every object it has seen is a CLIP embedding we search with `$vectorSearch`, and every attempt, map and observation is logged there.
> No training: words are matched to pictures with CLIP, and the library only helps — so it can go for objects it has never seen."

### 0:30–3:30 现场演示（Functionality）

操作：dashboard 顶部输入框输入一串（逗号分隔），按 **Grasp**。三块面板：命令（进度 item 2/3 和逐件 ✅/❌）· arm 相机（实时框）· 桌面 map（桶、物体、多久前看到）。

1. 输入：`yellow block, blue block, green block`（见 §2，最稳的三件）。
   > "It searches the table from above, picks the first view that shows the object, and drops it in the bucket. Watch the map on the right: every picture the arm takes updates it."
2. 第一件进桶后，**把桶挪到桌子另一边**（离底座 ≥25 cm，arm 从 z580 能看到）。
   > "Now I move the bucket. It never trusts an old position: before letting go it looks again — if the bucket isn't under the jaws, it climbs up, finds it, and follows."
   真机证据：桶不在观察视野时，系统从 (400,180,580) 找到桶 (437,299)，松手前确认偏差 0 mm（`eaeb298`/`2e23ac9`，照片 `docs/demo/bucket_moved_still_found.jpg`）。
3. 指着 map 底部的 "library: MongoDB Atlas · N views"：
   > "The object library lives in Atlas — this many views, searched with `$vectorSearch`. We keep a copy in memory for the inner loop, so perception went from 920 ms to 380 ms a frame, while Atlas stays the source of truth."
4. 时间够的话再输入一件新词（见 §2「评委给生词」）。

### 3:30–4:00 鲁棒性（Functionality）

> "If a grasp slips, it sees that in the picture and retries — up to three times, never switching to a different object. If a neighbour blocks a finger, it turns the wrist to a clear angle instead of giving up. The E-stop latches: nothing moves again until a person presses Reset."

### 4:00–4:30 模块化（Modularity）

> "It's one Viam module with a DoCommand API — `grasp`, `reset`, `status`, `preview`. The drop target is an environment variable: we switched from a taped square to this bucket in an hour. Grasp methods are picked by shape, not by name, and every new capability ships behind a flag."

### 4:30–5:00 收尾（Halloween）

> "Trick or treat: you name it, it bags it." 指南瓜桶。

---

## 2. 备用方案

### 选哪些物体（按真机记录）

| 稳（优先演示） | 证据 |
|---|---|
| yellow block | 进桶 24.1 s（`fc58103`） |
| blue block | 进桶 12.7 s（`3853386`）；进方框 18.0 s / 22.5 s，检查通过（`77bb53e`、`49f8b6c`） |
| green block / green cube | 进方框 27.5 s（`f3d12c1` 那轮），但 13:11 那次 servo 跟丢、把它推走了 |
| black tape | 进方框 2 次，20.1 s / 26.6 s，检查通过（`72ab6c5`、`466fd89`）；还没进过桶 |

| 不稳（别演示） | 原因 |
|---|---|
| 棒棒糖 | 夹糖球会滑；红棒棒糖出现过「报 done 但只是被推开」 |
| 美工刀（银 / 红） | 太薄，会滑；**银刀曾被误当成 "blue lollipop" 抓走**（13:21，待修） |
| M&M 袋、扁平零食 | 高度 < 10 mm，分割刚加上，没上真机 |
| 站立的圆柱 | 黄圆柱抬起时滑掉一次 |
| 水瓶 | 用户已决定不做 |

### 评委给生词时

- 先在 dashboard 输入，看它选中谁：会显示被认出的物体，或者拒绝原因。
  > "It's never seen this. CLIP matches the words to the picture; an open-set check refuses rather than grabbing a look-alike."
- 离线评测：没见过的名字 57/65 选对、2 次选错、6 次拒绝（改之前是 34 / 2 / 29，`05974ba`）；真机上 "banana" 搜了 4 个视角后拒绝，没抓任何东西。
- **诚实说明**：拒绝不是 100% 可靠，银刀被当成蓝棒棒糖那次就是反例。

### arm 出问题时

- **改讲录像或照片**：彩排时（14:30–15:00）用手机录一段 yellow block 和 blue block 进桶的完整视频。照片在 `docs/demo/`：
  - `bucket_found_from_above.jpg`：z580 一眼看到整个桶；
  - `bucket_with_yellow_block.jpg`：桶里是 yellow block 和一把刀；
  - `bucket_moved_still_found.jpg`：桶被挪到画面角落后，仍被找到。
- **Atlas 也可以讲**：打开 Atlas Data Explorer，`sorter` 库里有 `views`（库）、`attempts`（每次抓取）、`observations`（每帧）、`table_map`。

---

## 3. 操作清单

### 开机顺序

1. sfhack4 通电，网线插好（它掉过一次线，原因可能是断电或网线）。
2. 在 app.viam.com 看 machine 是 online；module `sorter-team_sorter` 正在运行。
3. 等 module warm：dashboard 状态显示 **ready · sees N objects**。
4. dashboard 在笔记本终端 tab「dashboard」里跑，浏览器打开 http://127.0.0.1:8000。**不要关这个 tab。**
5. **demo 前后都不要改机器配置**：保存配置会让 module 和相机重新加载，正在跑的任务会断。

### 桌面摆法

- 物体离 arm 底座 **≥ 25 cm**。更近的会被拒绝（下限 225 mm）。
- 物体之间间隔 **≥ 5 cm**：贴在一起的会被认成一块，或者挡住手指。
- 桶也放在 ≥25 cm 处，arm 从 z580 要能看到；桶口别离 arm 太远（> 约 60 cm 会报 `bucket_out_of_reach`）。
- 物体别贴着桶。arm 自己的线缆别垂在镜头前。

### 急停

- 按下：arm 立刻停，dashboard 显示 **E-STOP**，所有动作命令都会被拒（`77bb53e`）。
- 恢复：**拧开急停按钮 → 在 dashboard 点 Reset**。Reset 会解锁并清除错误。
- arm 停在低处时，先让 coordinator 竖直抬起再回观察位，**别直接平移**。

---

## 4. 做了什么（评委问答用）

### 真机验证过

| commit | 内容 |
|---|---|
| `3853386` / `eaeb298` | 放进桶：抓起 → 升到桶沿上方 → 松手。只用 5 s 内看到的桶位置；松手前再确认，看不到就升高找。blue block 12.7 s 进桶；桶被挪出视野后仍被找到，确认偏差 0 mm |
| `9b65b87` / `d1ce088` | 找桶（橙色 + 桶沿拟圆，按桶沿高度算 xy）：z580 和 z400 两个视角相差 1.5 mm；桶写进 live map |
| `d6a6897` | 手指落点被挡时转腕：真机日志 "turned the jaws −30 deg" |
| `72ab6c5` | 夹爪最低 z 174：两次 overcurrent 都发生在 z171，加上这个下限后没再撞过 |
| `49f8b6c` | 用画面判断是否滑落（夹爪传感器两个方向都会错）：清方框时真滑掉的两次都识别出来，第三次成功 |
| `77bb53e` | 急停锁定 + 竖直恢复（真机上被急停后，手动恢复过） |
| `743e816` | Atlas library 在内存里镜像，`$vectorSearch` 在后台：每帧 918 → 380 ms |
| `4efac3e` / `fdaae28` | Atlas 后台同步；断电后 JSONL 里的坏行不再让 module 停止响应 |
| `f306c36` | 一句话多件：真机解析并逐件执行，有部分成功（2/3） |
| `05974ba` | 开放词表：真机上 "banana" 搜完后拒绝，没抓任何东西 |
| `af761d7` | dashboard 三块实时面板，浏览器里看过 |
| arm 配置 | 150°/s、加速度 220：起停更柔和，100 mm 移动速度不变（实测） |

### 离线测过、真机还没验证 / 还不稳

| commit | 内容 |
|---|---|
| `2e23ac9` | 重试不换成第二名（但 13:21 仍发生过把刀当成蓝棒棒糖） |
| `fc58103` | 跟踪跳变 > 3 倍就停，不推着东西走 |
| `17c13e7` | 覆盖率搜索（40 mm 格子，贪心选视角） |
| `42b4587` / `76803bb` / `2c278a1` | 倾斜视角的相机模型（倾 25°，误差约 5 mm）；搜索还没接上 |
| `074dba1` / `178feec` | 扁平物体分割和抓法 |
| `ef4a60b` | 胶带内撑抓法（默认关） |
| `3857954` | 细标定脚本 `scripts/fine_calib.py`（还没跑） |
| `3825585` | gujunli 的 LLM 兜底（默认关，没有设 key） |

### 可能被问到

- **为什么用 Atlas，而不是本地文件？** 物体库和记忆跨重启、跨机器共享；`$vectorSearch` 按外观找最像的已知物体；抓取记录可以做统计，可以拿来学习。
- **为什么用 Viam？** 驱动、frame system、碰撞检查都现成；我们的代码是一个 module，换 arm 或换相机只改配置。
- **生词怎么办？** CLIP 直接比对文字和画面，library 只加分、不否决，再加一道 open-set 检验，分不清时宁可拒绝。
