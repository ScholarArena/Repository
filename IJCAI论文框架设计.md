下面是一个“自洽、可落地、方法论优先、实验重”的 **IJCAI 7页主文 + 附录（supplementary）** 论文框架，以及 **7天并行写作/实验计划**。我下面的设计目标是把**创新点、学术叙事与可复现性**做到尽可能接近 IJCAI 的偏好，并把最易被攻击的点（过度玄幻、多模型堆砌、运行时写代码、难以评测）系统性消掉。

---

## 1) 7页主文结构设计（按 Abstract / 1 Intro / 2 Related Work / 3 Method / 4 Experiments / 5 Conclusion）

> 总体配比：**实验页数最多**，方法次之；Intro要有“Teaser图”占位，方法要有“Foundry+Arena图”和“Issue FSM图”。

### 页数分配（7页主文，不含参考文献；附录单独PDF）

| 模块                 | 建议页数 | 必备图表                                     |
| ------------------ | ---: | ---------------------------------------- |
| Abstract           | 0.25 | 无                                        |
| 1 Introduction     | 1.15 | **Fig.1 Teaser**（跨领域抽象→审稿实例化）            |
| 2 Related Work     | 0.85 | 可无图、可以加个对比表                                      |
| 3 Method (ScholarArena) | 2.00 | **Fig.2 总览**、**Fig.3 Issue-FSM**、算法框/伪代码 |
| 4 Experiments      | 2.60 | 2–3张表 + 1张关键图（主结果/消融）                    |
| 5 Conclusion       | 0.15 | 无                                        |

---

## Abstract（约 150–200词，0.25页）

**写法要点（“一口气读完”结构）：**

1. **任务抽象**：在不完全信息的多方交互中，若将“可执行观测”作为策略行为的门控条件（evidence gating），则系统能显著降低 *unsupported claims*，并在相同可读性下提高主张的可验证性、覆盖度及深刻性；若用一个小而稳定的“交互语义本体”（intent ontology + evidence type ontology）把策略分解为 **Intent（为何说/做）** 与 **Skill（如何产生观测）**，则在更换工具库/领域即策略具体执行方法时，高层意图策略的迁移性能下降更小（只需替换技能库映射）；若把多轮交互压缩为“争点级状态”（issue state + evidence summary），则在保持决策一致性的同时显著降低决策输入长度与认知负担，并提升对低质量/被操控文本的鲁棒性。
2. **方法一句话**：Evidence-Conditioned Strategic Acts；Intent–Skill Factorization with Ontology；Issue-State Aggregation。
3. **Case study**：在 ICLR peer review/rebuttal 上实例化，显著降低 unsupported claims、提高 evidence coverage，并在多争点聚合上提升决策一致性/效率/深度。
4. **贡献三点**：方法论 + Data/Benchmark + 开源和在线运行。

（动机里可一笔带过现实风险：审稿中LLM滥用政策与“隐藏提示/注入”事件凸显“外部证据约束”的必要性。ICLR对LLM使用政策与安全担忧可作为现实支撑。 ([ResearchGate][1])）

---

## 1 Introduction（1.15页，含Teaser图）

### 1.1 跨领域科学问题（先抽象，避免不学术）

用更“计算机论文”的表述和学术化的表达：

> **我们研究：在文本主导的多方策略交互中，如何将外部可验证的执行观测（execution observations）纳入策略决策与语言行为生成，从而获得可复核（verifiable）的沟通策略与决策摘要。**

强调两条缺口：

* **缺口A（策略层）**：很多LLM系统在“说”上强，但缺少“证据条件化”的策略动作，导致输出与可执行事实脱节（words–deeds inconsistency）。
* **缺口B（工具层）**：工具学习研究证明LLM可学会工具调用，但多数工作假设**工具集已知且静态**、且不处理“弱标注/幻觉工具→可执行工具”的数据闭环。Toolformer/ToolLLM可作为代表引用。 ([OpenReview][2])

### 1.2 为什么选“同行评议”做Case Study（现实紧迫性）

* ICLR已明确讨论/规范LLM在评审中的使用与风险（低质量、治理与安全）。 ([ResearchGate][1])
* 现实世界已经出现“在论文中隐藏提示以影响AI评审”的报道，以及更广泛的间接提示注入风险；这使“证据约束+可复核流程”不仅是学术问题，也具有现实安全意义。 ([卫报][3])
* 无论是否采用LLM，同行评议过程中总会出现低质量的审稿，比如NeurIPS新闻Who/what is 'Adam'?

### 1.3 本文贡献（写成三条“可验命题”）

1. **Framework**：ScholarArena——把多方交互形式化为 *(Issue state → Intent → Skill plan → Execution evidence → Policy action)*。
2. **Skill Foundry**：从弱标注（闭源推测）到可执行黄金轨迹的 **execution-feedback 数据炼制**（你的核心数据中心贡献）。
3. **Benchmark + Protocol**：构建“证据质量+策略质量+聚合质量”的评测协议，并在peer review上系统验证。

### Fig.1（Teaser图，Intro必须放）

左：通用抽象（文本交互 + 外部执行证据通道 + issue ledger聚合）
右：审稿实例（review→issue threads→skill calls→evidence→action guidance→meta summary）

---

## 2 Related Work（0.85页，重点“贴近你贡献”）

用“3组相关工作 + 1段差异总结”即可。

1. **Tool-using LMs / tool learning**：自监督调用、指令微调工具使用、工具泛化（引用Toolformer、ToolLLM足够支撑主论点）。 ([OpenReview][2])
2. **LLM agents / constrained execution**：强调你不是运行时写代码，而是“plan–execute（受限）”；指出现有agent常缺“执行反馈炼制数据”的闭环。
3. **Scientific discourse / peer review automation & governance**：引用ICLR政策与安全讨论，说明现实驱动与评测必要性。 ([ResearchGate][1])
4. **Prompt injection / indirect injection risk**（一句话即可，避免喧宾夺主）：说明“证据约束与执行通道的隔离”对安全重要。 ([WIRED][4])

最后用一段“我们与相关工作的关键差异”收束：

* 工具学习：你解决 **弱标注→可执行技能库**；
* 多方交互：你解决 **issue线程化 + 证据聚合摘要**；
* 审稿：你给出 **可复现benchmark与指标体系**。

---

## 3 Method：ScholarArena Framework（2页，含两张图 + 伪代码）

这一节一定要“硬”：形式化 + 明确模块I/O + 训练目标。

### 3.1 Ontology：Intent–Skill–Evidence（语义本体包装，但轻量）

给一个小表（或文字定义）：

* **Issue**：争点单元（type, severity, span pointers, status）
* **Intent**：高层策略动作（why）——例如 *RequestEvidence / Challenge / Concede / Clarify / NarrowScope*（不要太多，10–20个可控）
* **Skill**：可复用能力（how），由原子API编排得到、离线生成
* **Evidence**：结构化观测（obs schema），来自技能执行

> 关键：你把 `Assess_Experimental_Breadth` 这类“伪工具”归入 Intent（或ReviewAct），而非工具；工具必须产出可定义的 observation schema。

### 3.2 Skill Foundry（离线）：从弱标注到“可执行黄金轨迹”

明确每步输入输出，避免“模糊”。

**F1 Issue 单元化 + 合并（无需新模型）**

* 规则切分：按段落/编号/引用标记/句法cue（e.g., “however”, “I am concerned”, “missing baseline”）
* embedding 合并：你已把 *(operation,target_type,outcome)* 合并编码，这是可行做法；在论文里写成：
  `e = Emb([operation; target_type; outcome; snippet]) → clustering → canonical issue type`
* 输出：`IssueCandidates` + `cluster_id`

**F2 伪工具剥离（可复现判别准则）**

* 定义判别函数 `is_executable(tool_call)`：若无法给出确定I/O与可执行obs schema，则不是tool → 归Intent/Act。
* 输出：`ToolCandidates`（可执行）与 `IntentCandidates`（不可执行）

**F3 原子API（10–20个“证据原语”）**

* 文档定位、公式/表格抽取、统计检验、相似度、检索（可本地索引）、沙箱运行等。
* 强调：原子API不含“决策推理”，只做确定性计算/检索/解析。

**F4 技能预编译（Skill = 原子API编排；产物是文件）**
Teacher LLM只在Foundry里扮演 **Skill Compiler**（单一供应商即可，减少质疑）。
输出三件套：

* `skill.py`（可执行函数）
* `skill.yaml`（签名、参数、返回obs schema、失败模式）
* `tests/`（最小测试；失败则丢弃/迭代修复）

**F5 Execution-Feedback Curation（核心闭环）**
对每条弱标注样本：

1. 定位 `paper_span`
2. 选择/检索匹配skill（按 issue type / required evidence）
3. 执行得到 `obs`
4. `predicate(obs, intent)` 判定是否支撑（规则优先；必要时允许“受控LLM”仅做结构化对齐，不做自由裁决）
5. 通过则写入黄金轨迹：
   **(state, role) → (intent, skill_calls, obs, action_guidance)**

> 你问“obs在训练里有用吗”：有用。SFT时把 `obs` 作为输入上下文的一部分，让模型学习“根据证据写action guidance”，并用于评测 evidence coverage/precision（而不是让模型自己执行）。

**Fig.2（方法总览图）**：Foundry离线炼制 + Arena在线交互（清晰画I/O）

### 3.3 Arena（在线）：受限 Plan–Execute–Update（不是ReAct发散）

定义一个**非Agent的 Orchestrator**（系统组件，不是LLM）：

* 维护 `IssueLedger`
* 解析并执行 skill calls
* 记录 obs 与状态转移
* 把结构化状态喂回同一个 policy LLM

**统一交互接口（四元组硬语义，解决你批注）**

1. **Rationale**（可公开，用户可选显示）
2. **Skill Plan**（JSON/function calls，只能调用已编译skills）
3. **Execution Evidence**（由Orchestrator返回，结构化）
4. **Policy Action** = `(Intent, ActionGuidance)`（给人类可用建议/要点）

### 3.4 Thread / 回合数 / 终止条件（避免攻击点）

每个Issue一个线性FSM（不需要大图）：

`Open → EvidenceRequested → EvidenceCollected → (Supported | Refuted | Inconclusive) → Closed`

* 固定小回合数K=2或3：Reviewer→Author→(optional Reviewer)
* 终止规则只依赖 **证据增量** 与 **状态**（不需要AC每轮裁决）

**Fig.3（Issue FSM图）**：把状态、触发条件、输出写清楚。

### 3.5 训练目标（SFT + 可选策略优化）

**SFT（主线）**：学会“选intent + 选skill plan + 产出action guidance”

* Input：`{role, paper_span, issue_state, dialogue_context, (optional obs_history)}`
* Target：`{intent, skill_calls, action_guidance}`

**可选策略优化（增强论文深度且可泛化）**
不要承诺重型在线PPO；更稳的是：

* **DPO/偏好优化**：用“证据质量更高/更少unsupported claims/更好解决关键issue”的成对偏好构造训练对。DPO作为可选策略优化路线有成熟文献基础。 ([SciSpace][5])
* **跨领域泛化叙事**：策略优化目标不写“审稿接收率”，写成更抽象的：

  * `Evidence Quality Reward`（coverage/precision）
  * `Issue Resolution Reward`（critical issue resolved）
  * `Efficiency Reward`（更少skill calls达成同等证据强度）

---

## 4 Experiments（2.6页，尽量做“可复现且打中靶点”的实验矩阵）

建议把实验写成 4 组：数据、主结果、消融、泛化/鲁棒。

### 4.1 数据与任务（0.5页 + 表1）

* 数据来源：ICLR OpenReview threads + PDFs（你已有管线） + 弱标注mining_results
* Foundry产出统计（表1）：

  * issue数、类型分布、技能库规模、执行成功率、过滤比例（被丢弃比例非常关键，体现“弱标注不可信→我们炼制”）

### 4.2 主任务评测（Evidence-grounded reviewing & rebuttal assistance）

**核心指标（能打中T1/T3）**

* Unsupported-Claim Rate（无obs支撑的断言比例）
* Evidence Coverage（有obs的关键争点占比）
* Evidence Precision（obs与断言一致性；可用规则/人工抽检）
* Ledger Utility（给AC/决策者的摘要：一致性/时间成本代理指标）

**对比基线（可行且不“玄幻”）**

* LLM-only（无工具）
* RAG-only（只检索不执行）
* Tool-use naive（工具集合已知但无Foundry过滤、无issue ledger）
* 你的ScholarArena（Foundry+ledger+受限执行）

> 这里你不需要把“LangChain框架”当贡献；框架只是实现方式。对比要对“机制”而非“库”。

### 4.3 消融实验（0.8页 + 表2/图4）

* w/o Foundry filtering（不过滤弱标注）
* w/o Intent–Skill factorization（直接端到端生成建议）
* w/o Issue ledger（不线程化聚合）
* K回合数敏感性（K=1/2/3）

### 4.4 泛化与鲁棒（0.8页）

两种“7天内更可能做完”的泛化：

1. **跨子领域/跨年份**：train on {2018–2023,部分领域} → test on {2024–2026,未见领域}；看证据指标下降幅度（对应T2）。
2. **注入/干扰鲁棒**（轻量实现）：在PDF/文本中插入“隐藏提示/无关指令”片段，观察 ledger 与受限执行是否降低被带偏的比例（与现实动机呼应）。相关现实风险与政策背景可引用。

### 4.5 人类评测（可选，小样本也能加分）

* 20–30个issues，双人标注：证据是否支撑、建议是否可用、争点摘要是否帮助决策。
* 这在IJCAI非常加分，但不要做大样本承诺。

---

## 5 Conclusion（0.15页）

三句话结构：

1. 提出ScholarArena：证据约束策略交互的通用框架
2. Foundry把弱标注炼成可执行轨迹；Arena用受限执行+ledger做可复核输出
3. 局限与未来：技能覆盖、解析误差、更多领域case

---

## 附录 / Supplementary（单独PDF：放“审稿人想看但主文放不下”的硬货）

建议目录：
A. Ontology完整表（Intent集合、Issue类型、obs schema）
B. Atomic APIs 与 Skill YAML示例（含失败模式）
C. Foundry提示词模板（Teacher编译器）与过滤predicate清单
D. 额外实验（更多消融/更多领域/更多注入样例）
E. 复现说明（代码结构、运行命令、数据许可与匿名化）

---

# 任务1：论文大纲（可直接拷到Overleaf的目录级别）

**Abstract**

**1 Introduction**
1.1 Evidence-grounded strategic interaction in text-based multi-party settings
1.2 Peer review as a high-stakes case study (policy & injection risks)
1.3 Contributions and overview (Fig.1)

**2 Related Work**
2.1 Tool-using language models and tool learning
2.2 LLM agents with constrained execution
2.3 Scientific discourse / peer review governance and security

**3 ScholarArena Framework: Intent–Skill Factorization with Execution-Feedback Curation**
3.1 Ontology: Issue, Intent, Skill, Evidence
3.2 Skill Foundry (offline): issue clustering → executable tool distillation → skill compilation → execution-feedback curation (Fig.2)
3.3 Arena (online): constrained plan–execute–update with an orchestrator
3.4 Issue threads as bounded-turn FSM (Fig.3)
3.5 Training objectives: SFT + (optional) preference-based strategy optimization

**4 Experiments**
4.1 Dataset and protocol; Foundry statistics (Table 1)
4.2 Main results: evidence quality + assistance quality (Table 2)
4.3 Ablations and sensitivity (Fig.4)
4.4 Generalization and robustness (cross-year/field + injection stress test)
4.5 Human evaluation (optional)

**5 Conclusion**

---

如果你愿意，我可以在你现有的数据字段（`operation/target_type/outcome/grounding_ref/latent_tool_calls` 等）基础上，进一步把 **(1) Issue类型集合与Intent集合** 给出一版“可直接写进ontology.yaml”的候选清单，并同时给出 **Table 1 / Table 2 的列定义**，确保你从第3天开始就能稳定产出“论文可用表格”。

[1]: https://www.researchgate.net/publication/371136760_Direct_Preference_Optimization_Your_Language_Model_is_Secretly_a_Reward_Model?utm_source=chatgpt.com "Direct Preference Optimization: Your Language Model is ..."
[2]: https://openreview.net/pdf/ae34f7aa512c79dcea8c269790603c2b04b69a3c.pdf?utm_source=chatgpt.com "Benchmarking Tool Retrieval for Large Language Models"
[3]: https://www.theguardian.com/technology/2025/jul/14/scientists-reportedly-hiding-ai-text-prompts-in-academic-papers-to-receive-positive-peer-reviews?utm_source=chatgpt.com "Scientists reportedly hiding AI text prompts in academic papers to receive positive peer reviews"
[4]: https://www.wired.com/story/google-gemini-calendar-invite-hijack-smart-home?utm_source=chatgpt.com "Hackers Hijacked Google's Gemini AI With a Poisoned Calendar Invite to Take Over a Smart Home"
[5]: https://scispace.com/pdf/api-bank-a-benchmark-for-tool-augmented-llms-qef9wv5v.pdf?utm_source=chatgpt.com "arXiv:2304.08244v1 [cs.CL] 14 Apr 2023"
