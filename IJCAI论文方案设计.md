下面是一版期望“**自圆其说、可落地、叙事与方法都更像 IJCAI**”的方案。给出了 **Title + Teaser + Targets + Method (Foundry/Arena) + Training (SFT/RL) + Thread/FSM + Evaluation** 的完整闭环。同时把提到的 ICLR 现实事件（LLM-review 争议、隐藏 prompt/prompt injection、安全事件）融入“动机”而非工程八卦。叙事需要引用已检索到的近年资料支撑。

> 说明：会按 **IJCAI 最偏好的审稿口味：清晰科学问题 + 可复现实验 + 强基线 + 数据/方法贡献可迁移** 来优化。

---

## 1) 标题：精简但要和 Teaser 对齐

希望“短、学术、吸引力、与 teaser 一致”。建议目前为：
**ScholarArena: Evidence-Grounded Strategy Learning for Scientific Discourse**

保留 ScholarArena ，同时把“审稿、作者和编辑的工作”抽象成 *scientific discourse*，避免过窄。

---

## 2) IJCAI 风格 Teaser：跨领域科学问题 + 审稿是关键试验场

### 2.1 统一科学问题（需要学术的表述）

> **研究问题：在文本主导的多方交互与决策场景中（同行评议、法庭辩论、应急响应、对抗评估等），如何学习“受可执行外部观测约束”的策略性言语行为（strategic speech acts），使模型输出在语用上可读、在证据上可验证，并能在更换工具/技能库即执行方法时保持高层策略可迁移？**

核心挑战不是“生成像人的文本”，而是经典的 **words–deeds inconsistency**：模型能说得很像，但其关键主张缺乏任何外部可执行观测支撑；多轮交互中这种不一致会累积放大，尤其在工具使用、检索与推理链条被 prompt injection 影响时更严重。近期已经有针对“在同行评议/文档分析链条里用隐藏指令操控 AI 评审”的系统性研究与公开事件，凸显“可执行证据约束”的必要性。([卫报][1])

与此同时，工具学习方向（Toolformer / ToolBench / API-Bank / ToolLLM 等）证明了 LLM 能学会“何时调用什么工具”，但多数工作默认工具集合已知，且主要优化任务完成率，并未把“**观测—主张一致性**”作为中心目标，也未给出面向多方交互的结构化决策压缩。([arXiv][2])

### 2.2 为什么用 ICLR 同行评议做 case study（现实紧迫 + 数据可复现）

* ICLR 官方在 2025 年明确回应了 LLM 生成论文与 LLM 审稿质量/治理问题，并讨论了如何处理相关风险。([ICLR 博客][3])
* 近期还出现“隐藏 prompt 操控 AI 辅助评审/评阅”的新闻与论文，说明如果没有“可执行证据门控”，系统会被低成本攻击放大。([The Washington Post][4])
* 同时也有大规模实证研究/随机实验在探索 LLM 在评审流程中如何“辅助而非替代”，为你“负责任辅助”的定位提供现实依据（但你论文要把它写成 *evidence-grounded strategy learning* 的科学问题）。([adam.holter.com][5])

---

## 3) 三个科学靶点：先抽象成可检验命题，再落到科学交流的例子

为避免“太口语/太绝对”，我把靶点写成 **可检验的研究命题**（IJCAI 喜欢）。

### T1：Evidence-Conditioned Strategic Acts（证据条件化策略行为）

**命题**：在不完全信息的多方交互中，若将“可执行观测”作为策略行为的门控条件（evidence gating），则系统能显著降低 *unsupported claims*，并在相同可读性下提高主张的可验证性、覆盖度及深刻性。

* 审稿实例：强质疑（“公式错误/实验不显著/引用缺失”）必须绑定可执行观测（符号化简/统计检验/引用解析），否则只能退化为“请求澄清”。
* 指标（可量化）：Unsupported-Claim Rate、Evidence Precision、Evidence Coverage、Claim–Observation Consistency。

### T2：Intent–Skill Factorization with Ontology（语义本体约束的意图—技能因子分解）

**命题**：若用一个小而稳定的“交互语义本体”（intent ontology + evidence type ontology）把策略分解为 **Intent（为何说/做）** 与 **Skill（如何产生观测）**，则在更换工具库/领域即策略具体执行方法时，高层意图策略的迁移性能下降更小（只需替换技能库映射）。

* 审稿实例：同一 intent（检验显著性）在不同论文上只替换数据抽取与统计技能；intent policy 不需要重学 prompt。
* 指标：Cross-domain transfer drop（只换技能库/证据原语，不换 intent policy）；intent calibration（intent 与证据类型的匹配准确率）。

### T3：Issue-State Aggregation（争点状态汇聚的决策压缩）

**命题**：若把多轮交互压缩为“争点级状态”（issue state + evidence summary），则在保持决策一致性的同时显著降低决策输入长度与认知负担，并提升对低质量/被操控文本的鲁棒性。

* 审稿实例：Area Chair/Editor 不读全对话，只读每个 issue 的状态（Supported/Refuted/Inconclusive）、证据摘要、严重性。
* 指标：Decision consistency（与真实 meta-review/decision 的一致性）、Time/Token cost、Robustness under adversarial text（含隐藏 prompt/注入）。

---

## 4) 方法总览：Skill Foundry（离线）+ Arena（在线）

你的批注里最关键的要求是：**离线预生成技能，在线只规划与调用**。本节严格照此执行。

### 4.1 Skill Foundry：从弱标注推测 → 可执行黄金轨迹（核心贡献）

**输入**

* 真实数据：OpenReview 时间序列（review/rebuttal/meta-review/decision）+ 论文 PDF/LaTeX/markdown（你已有）。
* 弱标注：你用闭源模型反推的四元组片段（mining_results），但**不直接当真**。

**输出（黄金轨迹的“可训练”格式）**

* 不是让模型学写代码，而是学：
  **(context, role, issue_state) → (Rationale, Skill-Plan, Evidence, Policy-Action)**
* 其中 Evidence 是结构化观测，来自技能执行（而不是 LLM 想象）。其中 Policy-Action = Strategic intent + actionable guidance（面向人类）。

#### F1：Issue 单元化 + 同类合并（你问“embedding 合并 operation/target/outcome 可以吗？”）

可以，但要加两点“防伪”处理：

1. **候选切分（尽量不用 LLM）**

   * 基于 OpenReview review 的段落/编号/项目符号结构切分；
   * 基于引用锚点（“Eq.”、“Table”、“Fig.”、“Appendix”）与模式词（novelty/ablation/significance）做轻量规则归类。
2. **embedding 聚类合并（可用你说的三字段拼接）**

   * 表示向量：`op ⊕ target_type ⊕ outcome ⊕ (grounding_ref)`
   * 聚类：HDBSCAN（自动簇数）更适合噪声；KMeans 作为对照。

> 关键：这一步的目标不是“发现工具”，而是先把 **争点类型（issue types）** 稳定下来（novelty, correctness, significance, missing baseline, clarity, reproducibility, etc.）。

#### F2：语义本体（Ontology）建模：少概念但足够“学术封装”

你说想用“语义本体论包装”，又怕概念太多。建议只保留两层、各 8–15 个类即可：

* **Intent Ontology（策略意图本体）**：如 {Challenge, Request-Evidence, Suggest-Experiment, Concede+Patch, Refute+Counterevidence, Clarify, Summarize, Decide}
* **Evidence-Type Ontology（证据类型本体）**：如 {Symbolic, Statistical, Citation, Implementation, Figure, Dataset, Prior-Art}

本体的获取方式写成可复现流程：

* 先由聚类簇中心 + 少量人工（或 teacher LLM 一次性）命名；
* 再用一致性检查（inter-annotator agreement / self-consistency）验证稳定性。
  这样既“学术”，又不会被喷“拍脑袋造概念”。

#### F3：原子工具不是从 latent_tool_calls 直接读出来，而是“证据原语集合”

你批注非常关键：latent_tool_calls 里很多是“伪工具”。因此定义一个 **Executability Criterion**（可执行判别准则）：

> 若一个条目能被映射为**确定性 I/O** 的 evidence primitive，并能在给定 paper artifact 上产生结构化观测，则它是 tool/primitive；否则它是 intent/act。

据此你只需要 10–20 个确定性 primitives（示例）：

* 文档定位：extract_span / extract_equation / extract_table / figure_crop
* 引用与检索：resolve_citation / local_scholar_search（建议本地索引，避免线上不可复现）
* 统计：t_test / bootstrap_ci / effect_size
* 符号：sympy_simplify / dimensional_check
* 代码：sandbox_run（严格隔离）

#### F4：技能预编译（Skill = primitives 的可复用编排；不和“agent技能”概念混淆）

你担心“技能概念混淆”，写法上这样处理最安全：

* 把 **Skill 定义为 deterministic program template**：
  `skill := (schema, planner-visible signature, executable implementation, tests)`
* 把闭源 LLM 的角色明确为 **Compiler / Distiller（离线编译器）**，不参与在线决策。

产物落盘（满足你要“文件 + 描述 + 可调用”）：

* `skills/<name>/skill.py`（实现）
* `skills/<name>/skill.json`（参数、返回 evidence schema、失败码）
* `skills/<name>/tests/`（最小可运行测试）

> LLM 只做“把弱标注编译成可执行技能”的一次性工作；在线决策不依赖它。

#### F5：Execution-Feedback Curation（你要的“迭代生成→执行→反馈→丢弃”闭环）

对每条弱标注样本（issue）：

1. 由 grounding_ref 定位 paper span
2. 选择对应技能（或技能候选集）执行，得到 observation（结构化）
3. 用 **predicate 优先** 的方式判定 observation 是否支持该 intent

   * 例如显著性：p-value / CI / effect size 阈值
   * 引用缺失：bibtex key 是否可解析
   * 公式一致性：sympy 化简是否等价
4. predicate 不足覆盖的少数情况，可用 **同一个 teacher LLM** 做“结构化裁决”（只读 observation schema，不读长文本），保证 LLM 介入最小化。

这一步与现实风险高度对齐：ICLR 对 LLM reviews 的治理与近期隐藏 prompt 事件都说明“仅文本”非常脆弱，而执行反馈能把弱标注变成可复现证据。([ICLR 博客][3])

**你问：Observation 在训练里有没有用？**
有，而且是这篇论文“不像工程”的关键：你可以把学习分解成两个监督子任务（都可复现）：

* **Plan task**：输入 context，输出 intent + tool_plan（此时还没 observation）
* **Act task**：输入 context + observation，输出 policy_action（行动指导）
  这比“让模型自己脑补证据”更科学、更可测。

---

### 4.2 Arena：在线只做“受控规划与调用”，不写代码、不 ReAct 发散

**核心循环（形式化写法，便于 IJCAI）**
对每个 issue 线程，在时间步 t：

* 状态 (s_t = (role, paper_span, dialogue_summary, issue_state))
* 策略输出动作 (a_t = (intent_t, plan_t))
* 环境执行：(o_t = \text{Exec}(plan_t))（结构化 observation）
* 策略输出：(u_t = \text{PolicyAction}(intent_t, o_t))（给人的行动指导）
* 状态更新：issue_state 依据 (o_t) 与对话动作更新（FSM）

#### Tool-Plan 表达：Function Calling / JSON（你说你没经验，我给“不会翻车”的写法）

* 采用**严格 JSON Schema**（或 function calling）输出 plan：工具名、参数、依赖、重试策略。
* 不允许自由文本混入 plan（否则可注入、不可复现）。
* 相关路线可在 Related Work 中对齐 Toolformer / ToolBench / API-Bank / ToolLLM；你做的是把它们推进到“证据门控的策略交互”。([arXiv][2])

#### Orchestrator 是什么（你问“是不是一个 Agent”）

* **不是 Agent**。它是确定性的系统组件（Python 控制器）：

  * 校验 plan 合法性（schema/权限/资源）
  * 调度技能执行（sandbox/timeout）
  * 返回 observation + failure code
  * 记录可复现日志（用于评测与复盘）

你可以在论文里把它叫 **Execution Controller**，避免“又一个智能体”引入概念膨胀。

#### Policy-Action 输出（补齐 Intent 后的“可用指导”）

Policy-Action 不限制动作空间，只要求输出结构化、可落地：

* Reviewer：**claims + evidence + requested changes**（不是全文 review）
* Author：**response strategy + required evidence/tasks list**
* Editor/AC：**issue summary + risk flags + decision suggestion**

“Policy-Action 旨在输出可执行的评审/回复/决策骨架（evidence-grounded action guidance），而非替代人类撰写完整评审文本；最终文本可由人类根据该骨架自行组织。”

---

## 5) 训练：SFT 为主线，RL/偏好优化为加深与泛化

### 5.1 SFT：训练目标是“选技能 + 产出行动指导”，不是写可运行代码

训练数据来自 Foundry 的黄金轨迹（两段式）：

1. **Plan-SFT**
   Input: (s_t) → Target: (intent, tool_plan)
2. **Act-SFT**
   Input: (s_t, o_t) → Target: policy_action

这直接回应你批注：“让开源模型写能跑通的代码不现实”。

### 5.2 RL/策略优化：如何得到“AC 采纳/接受概率”（不靠玄学）

你担心“AC 采纳率怎么来”。可行且学术的做法是 **Outcome Proxy Model**：

* 用真实 OpenReview 标签训练一个 (f_\phi)：
  输入：issue ledger（每个 issue 的 evidence strength、状态、严重性、是否解决）
  输出：(P(\text{accept})) 或 (P(\text{meta-review supports reviewer on issue}))
* RL/DPO 的 reward 用：

  * evidence success / unsupported claim penalty
  * critical issue resolved count
  * (\Delta f_\phi)（让预测接受概率上升）

如果你想更 IJCAI、少工程重型：主线可用 **DPO/偏好学习**，把“更好的 policy_action”（证据更充分、成本更低）作为偏好对训练；DPO 是近年常用稳定路线。([arXiv][6])

### 5.3 “泛化不只审稿”的写法（你要求抽象顶层设计）

你把问题定义为“Evidence-Grounded Strategic Interaction Game”，peer review 只是实例化。泛化实验不需要做“作战系统”，而是做**工具库替换**实验即可：

* 换成“法律检索 primitives”或“应急流程 primitives”，保留 intent ontology 与 policy；
* 做小规模迁移评测，报告 intent policy 的 drop。
  这样既体现通用性，又不跑题。

---

## 6) Thread/FSM：创建方式、回合数、终止判据（全部可复现、可防攻击）

### 6.1 Thread 创建：尽量传统方法，LLM 只做“轻量归类”

你批注问“能不能不用专门 Parser 模型”。可以：

* 先用规则切分（段落/编号/引用锚点）得到 issue candidates
* embedding 聚类合并同类
* 必要时用同一个 policy LLM 做“给 issue 打 type/severity 标签”（少输出类别、可控）

### 6.2 每个 issue 一个线性 FSM（不需要大图）

状态：`Open → EvidenceRequested → EvidenceCollected → (Supported | Refuted | Inconclusive) → Closed`

### 6.3 固定小回合 K（建议 2 或 3）

* K=2：Reviewer→Author
* K=3：Reviewer→Author→Reviewer（可选追问）
  超过 K 仍无新增证据 → `Inconclusive` 并交给人类/AC。
  这非常符合“辅助而非替代”。

### 6.4 “解决/僵局”判定：规则优先 + 证据增量

* 若 observation 支持 reviewer claim，且 author 未给出反证 → Supported
* 若 observation 反驳 reviewer claim，且 reviewer 无新证据 → Refuted
* 若连续两轮 observation 无增量 → Inconclusive
  AC/Editor 只在最后读 ledger 汇总，不需要每轮裁判。

---

## 7) 论文贡献（按 IJCAI 习惯写成 3 条）

1. **Framework**：提出 *Evidence-Grounded Strategy Game*：把多方文本交互形式化为 **intent 选择 + skill 调用 + 执行观测 + policy action**，并以本体约束实现 intent-skill 因子分解与迁移。
2. **Skill Foundry**：提出从弱标注推测到可执行黄金轨迹的 **Execution-Feedback Curation** 管线（技能预编译、可执行判别准则、predicate/结构化裁决）。
3. **Benchmark/Case Study**：在 ICLR peer review 上构建可复现基准与评测协议，并覆盖现实威胁（低质量 LLM reviews、隐藏 prompt/prompt injection）。([ICLR 博客][3])

---

* [The Washington Post](https://www.washingtonpost.com/nation/2025/07/17/ai-university-research-peer-review/?utm_source=chatgpt.com)
* [卫报](https://www.theguardian.com/technology/2025/dec/06/ai-research-papers?utm_source=chatgpt.com)
* [时代报](https://www.thetimes.com/uk/science/article/ai-ethics-guide-citations-nsnjmz25b?utm_source=chatgpt.com)

[1]: https://www.theguardian.com/technology/2025/jul/14/scientists-reportedly-hiding-ai-text-prompts-in-academic-papers-to-receive-positive-peer-reviews?utm_source=chatgpt.com "Scientists reportedly hiding AI text prompts in academic papers to receive positive peer reviews"
[2]: https://arxiv.org/abs/2302.04761?utm_source=chatgpt.com "Toolformer: Language Models Can Teach Themselves to Use Tools"
[3]: https://blog.iclr.cc/2025/08/26/policies-on-large-language-model-usage-at-iclr-2026/?utm_source=chatgpt.com "Policies on Large Language Model Usage at ICLR 2026"
[4]: https://www.washingtonpost.com/nation/2025/07/17/ai-university-research-peer-review/?utm_source=chatgpt.com "Researchers are using AI for peer reviews - and finding ways to cheat it"
[5]: https://adam.holter.com/hidden-prompt-injection-attacks-in-arxiv-papers-threats-to-ai-powered-research-tools/?utm_source=chatgpt.com "Hidden Prompt Injection Attacks in arXiv Papers - Adam Holter"
[6]: https://arxiv.org/abs/2305.18290?utm_source=chatgpt.com "Direct Preference Optimization: Your Language Model is ..."
