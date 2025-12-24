# DDXPlus CausalQA：用病例统计给 `more/less` 提供“依据”

这份说明对应脚本：`01/build_ddxplus_causalqa.py`（`--mode stats`）。

## 为什么要做“统计版”的 `more/less/no effect`

原来的 LLM-only 版本让模型同时决定：
1) 题目怎么写（`question_stem`），以及  
2) 答案方向（`more/less/no effect`）。

这样做的问题是：`more/less` 很可能只是“模型的常识猜测”，难以解释“为什么是 more / 为什么是 less”。

统计版的核心改动是：**让 `answer_label` 由 DDXPlus 病例数据的观测统计决定**，LLM（可选）只负责把“疾病 + 一个证据描述”写成自然语言题干。

## DDXPlus 里有哪些可用信息

DDXPlus（本项目中路径 `01/Dataset/DDXPlus/22687585/`）至少提供三类文件，分别服务于：

1) **疾病→相关证据集合**：`release_conditions.json`  
   - 每个疾病有 `symptoms` 和 `antecedents` 两个 evidence-code 集合（键为 `E_XXX`）。
2) **证据码→自然语言问题**：`release_evidences.json`  
   - 给每个 `E_XXX` 的 `question_en`（以及 `is_antecedent`, `data_type` 等元信息）。
3) **病例→真实疾病 + 出现过的证据**：`release_train_patients`（以及 validate/test）  
   - 关键字段：`PATHOLOGY`（真值疾病），`EVIDENCES`（该病例出现的证据列表）。

## 统计流程是什么（我们到底算了什么）

定义：
- `D`：某个疾病（例如 `Pneumonia`）
- `E`：某个证据码（例如 `E_91`，其 `question_en` 是发烧相关问题）

从病例表中一次遍历统计四个计数：
- `N`：总病例数
- `nD`：`PATHOLOGY == D` 的病例数
- `nE`：证据 `E` 在 `EVIDENCES` 中出现的病例数
- `nDE`：同时满足 `PATHOLOGY == D` 且出现 `E` 的病例数

然后计算两个条件概率（**观测相关性**）：
- `P(D|E) = nDE / nE`
- `P(D|not E) = (nD - nDE) / (N - nE)`

最后用差值做方向判定：
- `diff = P(D|E) - P(D|not E)`
- 若 `diff > τ` → `answer_label = "more"`
- 若 `diff < -τ` → `answer_label = "less"`
- 否则 → `answer_label = "no effect"`

脚本里：
- `τ` 对应 `--tau`（默认 `0.005`）
- `nE` 的最小支持度对应 `--min-evidence-support`（默认 `200`）

### 一个具体例子（肺炎 + 发烧）

在 `release_train_patients` 上，脚本同样的计算会得到类似结果：
- `D = Pneumonia`
- `E = E_91`（fever）
- `P(D|E)` 明显大于 `P(D|not E)`，因此 `diff > 0` → `answer_label="more"`

这就是“为什么它是 more”：**因为在训练病例里，出现发烧证据时肺炎的后验概率更高**。

## 题目是怎么生成的（拿到疾病以后用什么信息）

统计版每道题都会先为该疾病选一个证据码 `E`：
1) 从 `release_conditions.json` 里取该病的 `symptoms/antecedents` evidence-code 候选集  
   - 由 `--evidence-pool` 控制（`symptoms/antecedents/both`）
2) 用上面的 `diff` 对候选证据排序（默认按 `|diff|` 强弱排序，并考虑支持度）
3) 在每个 disease 的 Top-K 里抽一个（`--top-k-per-disease`）用于多样性

题干（`question_stem`）生成时使用的信息只有：
- 疾病名（`get_disease_label(disease)`）
- 证据的英文问题描述（`release_evidences.json` 的 `question_en`）

实现上：
- 默认会尝试用 LLM 把 `question_en` 改写成更自然的一句话（可用 `--no-llm-question` 关掉）
- 关闭 LLM 时，会用一个确定性的单句模板兜底：
  - `If the patient answers YES to "<question_en>", how will this affect the probability of <Disease>?`

## 为什么会出现 “more 特别多”

你在 `--mode stats` 下看到 `more` 占比很高，一般有两个原因：

1) **候选证据来自该疾病的 `symptoms/antecedents` 列表**：这些证据本身就是为了“诊断”而设计的特征集合，出现在病例里时通常会让该病的后验概率变大（`diff > 0`），于是更容易被判成 `more`。
2) **证据码是“正例证据”**：`EVIDENCES` 里出现的 `E_XXX` 往往代表“该证据成立/出现了”，而不是“该证据缺失”。在这种编码方式下，负相关（`diff < 0`）的证据天然会比较少。

## 如何做到 `more/less` 各一半

严格靠 `diff < 0` 来产生大量 `less`，通常做不到（因为负相关证据少）。因此脚本提供了一个更直接的办法：

- 对同一个证据 `E`，既可以问“病人回答 YES”（对应 `E` 成立），也可以问“回答 NO”（对应 `E` 不成立）。
- 关键点：**不是“NO 本身就等于 less”**，而是我们已经用统计确定了“`E` 成立 vs 不成立”哪个更支持疾病：
  - 若 `P(D|E) > P(D|not E)`，则“`E` 成立”对应 `more`，“`E` 不成立”对应 `less`；
  - 若 `P(D|E) < P(D|not E)`，则相反。
- 题干可以写成两种等价风格：
  - 状态：`If the patient has ...` / `If the patient does not have ...`
  - 变化：`If <finding> develops` / `If <finding> resolves`

这会把 `less` 变成“E 的缺失/缓解”带来的方向变化，从而在不依赖负相关证据的情况下做到 50/50：
- 使用：`--balance-more-less --target-more-ratio 0.5`
- 同时会自动排除 `no effect`（只保留 `more/less`）。

## 这算“因果”吗？需要 `do()` 才是真因果吗？

这里算出来的是 **观测相关性**（association），严格来说不是因果效应：
- 我们用的是 `P(D|E)` vs `P(D|¬E)`（条件概率差）
- 真正的“因果效应”通常写作 `P(D | do(E=1))` vs `P(D | do(E=0))`

要从观测数据得到 `do()` 分布，一般需要：
- 明确干预变量 `E` 的含义（可操作的“处理/干预”）
- 有足够的协变量用于控制混杂（confounders）
- 采用可识别的因果假设（例如后门调整），或者有随机试验/自然实验

在 DDXPlus 里尤其需要小心：
- `symptoms` 很多是疾病的**结果**而不是原因（例如“发烧”更像肺炎的表现），这时把它当 `do(E)` 的干预变量在因果上并不合理；
- `antecedents` 更接近风险因素，但仍可能被年龄、性别、其他既往史等混杂。

所以结论是：
- **如果你追求严格因果（干预语义）**：需要 `do()` 框架 + 识别假设/调整方法，单纯的 `P(D|E)` 不够。  
- **如果你的目标是做“数据驱动的 WIQA 风格方向题”**：`P(D|E)` 的差值可以作为一个可解释的、可复现的“依据”（但它是 association，不是因果保证）。

## 下一步怎么做得更“像因果”

如果你想往 `do()` 靠近，可以考虑：
- 只用 `is_antecedent=True` 的证据码当作候选（更接近可干预的风险因素）
- 用 `AGE/SEX` 等基础变量做调整（例如逻辑回归/倾向评分加权/匹配）
- 明确一个因果图（DAG）后做后门调整（需要你决定哪些变量是混杂、哪些是中介/碰撞点）
