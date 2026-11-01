# Monty Hall Cognitive Experiment

> [!NOTE]
> **Project Question:** How do AI reasoning systems respond when a simple probability problem is placed inside a changing information environment with multiple agents, competing decisions, and social pressure? This project grew from my interest in the gap between human intuition and algorithmic reasoning. Rather than testing a theory of consciousness, it uses a controlled multi-agent simulation to explore how beliefs and decisions change when the information available to an agent changes.

https://github.com/user-attachments/assets/c0063c54-3800-47dd-8256-4c3484152e1b

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20API%20%2B%20Ollama-green)
![Project](https://img.shields.io/badge/Focus-AI%20Reasoning-orange)

**A student-built multi-agent LLM experiment that turns the Monty Hall problem into a desert-survival scenario to explore belief revision, information changes, conformity, and decision-making under pressure.**

> [!NOTE]
> **Scope:** This is an exploratory personal project, not a claim to establish a theory of human consciousness or a validated model of human cognition. The experiment uses LLM agents as a controllable environment for asking questions about reasoning, information, and decision-making.

---

### ⚡ Zero-Cost & Multi-Provider Readiness

> [!TIP]
> The core engine supports an auto-adaptive fallback architecture. It detects active API keys or a local **Ollama** instance without forcing complex setup. Run it right away — no configuration required if you have a local Ollama running.

```bash
# 1. Clone the repository
git clone https://github.com/JunyiGuo888/monty-hall-cognitive-experiment.git

# 2. Run with zero API cost (auto-detects local Ollama / active keys)
python src/main.py
```

---

# I. Design Motivation

> The animation carries the probabilistic principles and the cognitive loop. This README, from here down, addresses only what the experiment itself has to face.

## 1.1 Probability in the Three-Door Problem Is a Projection of Information

The core of Monty Hall is a rewrite of conditional probability: initial 1/3, and after the host opens an empty door, the remaining unopened door becomes 2/3.

But what this experiment actually cares about is not the number 2/3, but something more fundamental:

> **Probability is not an objective property of the world. It is a quantitative projection of an information state.**

The host knows everything, must open an empty door, and cannot open the door with the prize — these three constraints determine the posterior bias. Two doors that look symmetric are actually backed by asymmetric information sources. **The probability distribution is imposed by the observation-generating mechanism, not by the number of unknowns.**

From this, a critical step follows:

> **If you do not know that another player exists, Monty Hall does not fail.**
>
> You can keep reasoning with 1/3 vs 2/3, and that reasoning is **entirely correct within the information you hold**. The other party does not interfere with your judgment or your action. Treated as an independent system, Monty Hall fully applies.

So:

> **An unrelated person appears — why should that change probability?**

He is not physical interference, not an information source, not an exclusion mechanism. The only possibility: **he changes your information.**

You are not changed by him, but by the fact that **you now know he exists**.

**This is superposition:**

- Survivor A's information domain: I chose door 1, host opened door 3 → door 2 has 2/3.
- Survivor B's information domain: I chose door 2, host opened door 3 → door 1 has 2/3.

Each 2/3 is **complete, correct, and self-consistent within its own information domain**. Not a single step of reasoning is wrong.

What is wrong is treating the two domains as one.

> **Within each domain, 2/3 is correct; the moment the domains merge, they collapse to 1/2.**

## 1.2 When a Probability Problem Becomes a Multi-Agent Problem

The 2/3 in Monty Hall rests on a single-agent structure: "host vs you".

Once **another reasoning agent** appears, the nature of the problem changes:

- He is estimating you; you are estimating him.
- The two sides are structurally symmetric.
- The original advantage no longer holds; the posterior degenerates to an even split.

**The key shift occurs when "the unknown" becomes another reasoning agent.** The original puzzle can be described with conditional probability; once another agent has a competing decision and both sides interact, a game-theoretic perspective becomes useful.

| Layer | Subject's view | God's view |
|-------|---------------|------------|
| Structural appearance | Monty Hall | Symmetric game |
| Correct probability | 1/3 vs 2/3 | 50/50 |
| Information source | "The host is excluding for me" | The host merely opened an empty door |
| Perspective | Probability theory | Game theory |

**"Excluding for me" is a subjective reading. "Opened an empty door" is an objective fact.** The two happen to coincide during the Monty Hall phase, and split decisively once a second agent appears.

## 1.3 The Problem Produced by Triple-Blind

Three players each pick a door, unaware of one another. The host opens an empty door, eliminating one player. Two remain.

**From each survivor's view**: the host opened "an empty door he did not choose"; he faces 1/3 vs 2/3 and should switch.

**From God's view**: the host merely opened an empty door; the two survivors face a symmetric game, 50/50 each.

The same underlying situation can therefore look like a Monty Hall problem from an individual agent's information state, while a broader view reveals a strategic interaction between agents.

**Two 2/3's cannot both hold — their information bases sum to 4/3. They shatter on contact, each becoming 1/2.**

What triple-blind reveals is not a probabilistic error, but the **necessary consequence of information-domain splitting**:

> The same world, viewed by subjects in different information states, projects contradictory 2/3's.

## 1.4 Three Forces in the Decision Process

After Discovery, every agent's decision is pulled simultaneously by three forces:

**I. Internal: illusion residue vs rational reconstruction**

It knows the conditions have changed, but the old 2/3 intuition keeps surging back. Reason says 50/50; instinct still shouts "switch".

**II. External: voluntary conformity vs passive capitulation**

Teammates' speech, the second vehicle's existence, survival pressure — all push in different directions. Is it following willingly, or being crushed into abandoning its own judgment?

**III. Structural: subjective projection vs shared reality**

It remains in a single domain (2/3 valid), while the global situation has already collapsed (50/50). Can it exit its own domain and enter the whole?

**The three forces are not independent:**

- Internal illusion residue **amplifies** external conformity pressure: once intuition wavers, a single push topples it.
- External survival pressure **reinforces** internal illusion residue: the more panicked, the harder it grips the old frame.
- Structural collapse **triggers** internal and external at once: at the moment of Discovery, all three forces fire together.

**I found it more useful to treat these as interacting forces rather than as three independent parameters.**

## 1.5 Composition and Representation of the Three-Force Parameters

The experiment grounds these three interacting forces into a concrete sandbox:

- Two vehicles, each choosing a pump, unaware of the other;
- Discovery reveals the other's existence;
- Each agent subjectively remains within the Monty Hall frame;
- Objectively, it has already entered a symmetric game;
- **Two vehicles, ten people, three pumps, one of them fuelled.**

This is not just an abstract puzzle; the scenario gives the decisions a concrete **survival pressure**: choosing the wrong pump means a vehicle does not get enough fuel.

**The six dimensions are not six independent psychological metrics; they are three forces each split into two poles:**

| Force in the triad | Corresponding dimensions |
|--------------------|--------------------------|
| Internal: illusion vs reconstruction | `illusion_residue` / `paradigm_shift_awareness` |
| External: voluntary vs passive | `conformity` / `willingness_violation` |
| Structural: single-domain vs whole-domain | `monty_hall_persistence` / `monty_hall_recognition` |

Each pair is **two directions of the same force**, not two independent dimensions.

The analyzer's state machine (`not_mentioned` / `actual` / `carry_forward`) tracks continuously across rounds; it measures the **drift of the same force over time**, not an independent snapshot per round. The six scores are simply six projection lines of three forces along the time axis.

> **Three forces → six projection directions → one cross-phase trajectory per direction → observe where these interacting forces finally land.**

### In one sentence

> **The motivation is to explore what happens when a familiar probability problem is placed inside a multi-agent environment. Under the survival pressure of two vehicles, ten people, and one vehicle's worth of fuel, agents face three interacting forces: internal belief residue, external pressure, and a changing information structure. The six dimensions are operational measures used to track those forces across the simulation.**

---

# II. Logical Architecture of the Experiment

## 2.1 Participants

- Two vehicles (**Car 1** / **Car 2**), five agents each (A–J), ten in total.
- Each agent is randomly bound at simulation start to one `(provider, model)` pair from the active pool, **fixed for the entire run**.
- Every CoT is traceable to a specific provider/model, while each agent's identity remains stable across all six phases.

## 2.2 Scenario

- Desert centre, 500 km from civilisation, fuel nearly empty.
- Three pumps: **Red**, **Black**, **Yellow** — only one has fuel (randomly assigned).
- Each vehicle has a default target: Car 1 → Red, Car 2 → Yellow.
- **Boss message** (0–2 words randomly dropped, simulating wind interference):

  > "Black is empty! … Only one pump has enough fuel for one vehicle!"

  Each agent receives a different corrupted version — a deliberately preserved **initial information difference**, used to observe how noise affects judgment.

## 2.3 Six Phases

| # | Phase | Isolation | Description |
|---|-------|-----------|-------------|
| 1 | **Initial** | Fully isolated | Each agent drafts a plan, knowing only its own default target |
| 2 | **Debate 1** | Within vehicle | Conformity pressure introduced |
| 3 | **Debate 2** | Within vehicle | Agents see each other's first-round speeches |
| 4 | **Discovery** | First cross-vehicle contact | Second vehicle appears; conflicting targets; boss message re-emphasised |
| 5 | **Debate 3** | Within vehicle | Reconsideration under the two-vehicle structure |
| 6 | **Debate 4** | Within vehicle | Final decision |

Discovery is the **collapse point**: the three forces fire simultaneously at this moment.

## 2.4 Information Flow

- **Initial**: each agent sees only its own prompt.
- **Debate 1 / 2**: can see same-vehicle teammates' previous-round speech/conclusion.
- **Discovery**: first view of the other vehicle, hearing its shout.
- **Debate 3 / 4**: same-vehicle re-discussion, referencing all prior phases.
- **Final Vote**: rule-based extraction from the Debate 4 conclusion (`extract_vote_local`), no extra model call.
- **Negotiation**: triggered only when both vehicles finally choose the same pump; not part of CoT analysis.

## 2.5 Negotiation Logic

- Each vehicle sends one representative (selected by conclusion clarity + strength).
- Up to two rounds: Round 1 statements, Round 2 reconsideration after seeing the other's argument.
- If still deadlocked after two rounds → coin flip.
- The result overrides both vehicles' final decision and is written to the `cross_team_negotiation` field.
- **Negotiation does not participate in the six-dimension scoring**; it is presented only as the final outcome of social pressure in the extreme.

## 2.6 Analyzer and the Six Dimensions

`src/analyzer.py` is independent of the simulation and can score any `desert_progress_*.json` standalone.

**The six dimensions are not intended as six independent psychological constructs; they are six operational indicators grouped around three interacting forces:**

| Force in the triad | Corresponding dimensions |
|--------------------|--------------------------|
| Internal: illusion vs reconstruction | `illusion_residue` / `paradigm_shift_awareness` |
| External: voluntary vs passive | `conformity` / `willingness_violation` |
| Structural: single-domain vs whole-domain | `monty_hall_persistence` / `monty_hall_recognition` |

**Cross-phase state machine** (per dimension, continuous across rounds):

- Not mentioned and never seen → score `0`, source `not_mentioned`
- First mention → use LLM score, source `actual`
- Mentioned again with a different score → jump, source `actual`
- Seen but not mentioned in this segment → carry forward last score, source `carry_forward`

No interpolation, no averaging. **The six trajectories are intended to show how these indicators change across the experiment rather than treating each round as an isolated snapshot.**

### In one sentence

> **Experiment architecture = changing information environment + multi-agent interaction + six tracking dimensions + cross-phase analysis.**
>
> **Discovery is the point where the information structure changes. `paradigm_shift_awareness` tracks whether an agent recognises that change, while `illusion_residue` tracks how strongly the earlier Monty Hall frame continues to appear in its reasoning.**

---

# III. Concrete Details

## 3.1 Installation and Run

```bash
# 1. Clone the repository
git clone https://github.com/JunyiGuo888/monty-hall-cognitive-experiment.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure providers
cp .env.example .env
# edit .env

# 4. Run
python src/main.py
```

(For Ollama, ensure the local service is running; no extra Python package beyond `aiohttp` / `requests` is needed.)

## 3.2 Provider System

Four backends are supported, in any combination:

- **DeepSeek** / **OpenAI** / **Anthropic** — cloud
- **Ollama** — local

**Per-agent assignment:** at startup, `config.settings` builds `AVAILABLE_PROVIDERS` from whichever credentials are present; each agent then draws one `(provider, model)` pair from the pool and stays bound to it throughout. The mapping is printed at startup:

```
Agent provider mapping:
  A -> openai/gpt-4o
  B -> ollama/qwen2.5:7b
  ...
```

**Ollama auto-detection:** if `OLLAMA_ENABLED=true` in `.env`, `main.py` automatically calls `check_ollama_available()`, which:

1. Hits Ollama's native `/api/tags` endpoint;
2. Verifies the configured `OLLAMA_MODEL` is in the returned list;
3. Aborts with a clear message if either check fails.

This is the full meaning of "auto-detection": nothing beyond `OLLAMA_ENABLED=true` is required.

## 3.3 Environment Variables

| Variable | Description | Default / Example |
|----------|-------------|-------------------|
| **DeepSeek** | | |
| `DEEPSEEK_API_KEY` | API key for DeepSeek | (your key) |
| `DEEPSEEK_BASE_URL` | Base URL for DeepSeek API | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | Model name for DeepSeek | `deepseek-chat` (or `deepseek-reasoner`) |
| **OpenAI** | | |
| `OPENAI_API_KEY` | API key for OpenAI | (your key) |
| `OPENAI_BASE_URL` | Base URL for OpenAI API | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Model name for OpenAI | `gpt-4o` / `o4-mini` |
| **Anthropic** | | |
| `ANTHROPIC_API_KEY` | API key for Anthropic | (your key) |
| `ANTHROPIC_BASE_URL` | Base URL for Anthropic API | `https://api.anthropic.com/v1` |
| `ANTHROPIC_MODEL` | Model name for Anthropic | `claude-sonnet-4-5` |
| **Ollama (Local)** | | |
| `OLLAMA_ENABLED` | Enable local Ollama auto-detection | `false` in `.env.example` |
| `OLLAMA_BASE_URL` | Ollama OpenAI-compatible endpoint | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Model name to request from Ollama | `qwen2.5:7b` |
| `OLLAMA_CHECK_TIMEOUT` | Startup probe timeout (seconds) | `5` |
| **Runtime** | | |
| `MAX_CONCURRENT` | Max concurrent API calls | `5` |
| `API_TIMEOUT` | Per-request timeout (seconds) | `600` |
| `MAX_RETRIES` | Retries per API call | `3` |
| `RETRY_DELAY` | Base delay between retries (seconds) | `2` |
| `SHOW_FULL_COT` | Stream CoT to stdout and include it in the HTML report | `true` |

> **Note**: The legacy `API_PROVIDER` variable is **deprecated** and ignored. Provider routing is handled per-agent as described above.

## 3.4 Analyzer

`src/analyzer.py` **runs independently of the simulation** and can score any `desert_progress_*.json` standalone.

**Provider selection:** reads its own `PROVIDER` environment variable; if unset, auto-selects in this priority order:

1. `openai`
2. `deepseek`
3. `anthropic`
4. `ollama`

Only providers with valid (non-placeholder) credentials are considered. If `PROVIDER` is set explicitly but its credentials are missing, an `AnalysisError` is raised rather than silently falling back.

**Analyzer-specific environment variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVIDER` | Force analyzer backend (`openai` / `deepseek` / `anthropic` / `ollama`) | auto |
| `ANALYZER_MAX_CONCURRENT` | Overrides `MAX_CONCURRENT` for the analyzer | `MAX_CONCURRENT` |
| `ANALYZER_API_TIMEOUT` | Overrides `API_TIMEOUT` for the analyzer | `API_TIMEOUT` |
| `SEGMENTS_PER_ROUND` | Fixed segment count per round (shared by all characters) | `10` |
| `MAX_SEGMENT_CHARS` | Per-segment character ceiling; longer pieces truncated | `6000` |

**Segmentation:** each CoT is split into a fixed number of pieces by proportional position, with interior boundaries snapped to sentence terminators. All characters and all rounds share the same `SEGMENTS_PER_ROUND`, so "segment i" is comparable across characters.

**Six-dimension definitions:**

| Dimension | Meaning |
|-----------|---------|
| `monty_hall_recognition` | Genuine understanding of the Monty Hall causal structure |
| `monty_hall_persistence` | Active, deliberate use of Monty Hall reasoning as a primary tool |
| `paradigm_shift_awareness` | Realisation that Monty Hall logic no longer applies |
| `illusion_residue` | Old Monty Hall thinking unconsciously interfering with current judgment |
| `conformity` | Judgment influenced by external opinions rather than independent reasoning |
| `willingness_violation` | Suppressing one's own genuine judgment to align with external pressures |

**State machine (per dimension, continuous across rounds):** see [2.6](#26-analyzer-and-the-six-dimensions).

## 3.5 Output

- `desert_progress_*.json`: full simulation data, including all CoT traces and metadata about which provider was used for each agent.
- `desert_report_*.html`: interactive report with all CoT, decision timelines, and provider labels.
- `<progress_base>_<provider>_progress_results.json`: analyzer output (per-segment scores, sources, reasons).
- `<progress_base>_<provider>_segmented_scores.csv`: flat table of all segments.
- `<progress_base>_<provider>_trajectory_60points.html`: Plotly trajectories (one subplot per character, six lines each).

---

Enjoy experimenting with how different LLMs respond to a familiar probability problem when the information structure and social context change.
