# Monty Hall Cognitive Experiment

> [!NOTE]
> **Core Hypothesis:** Human consciousness operates as an active, self-adaptive "codec" influenced by metaphysical intuition and curiosity, leading to systematic Bayesian cognitive biases when contrasted with algorithmic rationality. This simulation is designed to operationalise that hypothesis through controlled multi‑agent interactions, providing empirical ground for studying the gap between human‑like intuition and purely rational decision‑making.

![Monty Hall Simulation Dashboard](docs/montyhall.png "Experiment dashboard - animated GIF coming soon")

<!-- The above image placeholder will be replaced with an animated GIF in future releases -->

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20API%20%2B%20Ollama-green)
![Research](https://img.shields.io/badge/Field-Cognitive%20Science-orange)

**Multi-agent LLM simulation of a Monty Hall problem disguised as a desert survival scenario, designed to study Bayesian belief revision, paradigm shift, and the influence of subjective priors, strong illusions, conformity, and concession on AI reasoning.**

---

### ⚡ Zero-Cost & Multi-Provider Readiness

> [!TIP]
> The core engine features an auto-adaptive fallback architecture. It dynamically detects active API keys or local **Ollama** instances without forcing complex setup. Run it right away – no configuration required if you have a local Ollama running.

```bash
# 1. Clone the repository
git clone https://github.com/JunyiGuo888/monty-hall-cognitive-experiment.git

# 2. Run with zero API cost (auto-detects local Ollama / active keys)
python main.py

## Overview

This project provides a **controlled cognitive experiment** where multiple LLM agents (per vehicle) face a decision problem that is a **Monty Hall dilemma disguised as a desert fuel-pump choice**. The environment is engineered to induce a **paradigm shift**:

- Before any decision, agents receive a **noisy “boss message”** (Black pump empty, with partial word loss simulating wind interference).
- Initially, agents reason under classic Monty Hall (3 pumps, 1 with fuel → 1/3 vs 2/3).
- Later, a **second vehicle appears**, collapsing the probability structure into a symmetric two‑vehicle competition (50/50).

The simulation captures **complete reasoning chains (CoT)** for every agent at every phase, enabling researchers to analyze:
- How **subjective priors** (1/3 vs 2/3 heuristics) influence initial reasoning.
- How agents **revise beliefs** after the structural change (paradigm shift).
- **Residual Monty Hall illusions** – do they still apply 2/3 logic after the collapse?
- The impact of **social conformity** (within‑vehicle debates) and **concession** (cross‑team negotiation) on final decisions.
- **Environmental interference** – how partial information loss (boss message corruption) affects judgment.

All CoT data are saved in a structured JSON format, ready for external analysis. The framework features a **Dynamic Provider Dispatching** architecture, supporting **DeepSeek, OpenAI, Anthropic, and local Ollama** models simultaneously, enabling seamless comparative studies across different LLM families and local inference.

---

## Experiment Design

### Scenario
- Two vehicles (Car 1 and Car 2), each with 5 agents (A–J).
- Desert setting: fuel nearly empty, 500 km from civilization.
- Three pumps: **Red**, **Black**, **Yellow** – only one has fuel (randomly assigned).
- **Boss message** (delivered before the first decision, with 1–2 words randomly dropped):
  > “Black is empty! … Only one pump has enough fuel for one vehicle!”

### Phases (in chronological order)
1. **Initial Round** – each agent independently formulates a plan (isolated), having heard the boss message and knowing their vehicle’s default target (Car 1 → Red, Car 2 → Yellow).
2. **Debate 1 & 2** – agents discuss within their vehicle (social conformity pressure introduced).
3. **Discovery** – a second vehicle appears with a conflicting target; the same boss message is re‑emphasised, but the situation now involves two vehicles competing for a single pump.
4. **Debate 3 & 4** – agents reconsider under the new two‑vehicle structure.
5. **Final Vote** – each agent commits to a pump.
6. **Cross‑team Negotiation** (if both choose the same pump) – representatives from each vehicle negotiate for up to two rounds; if deadlock, a coin is flipped.

## Features

- Complete CoT (Chain of Thought) extraction from all models
- Modular design, environment-based configuration
- HTML report with detailed reasoning traces
- Cross-vehicle negotiation if both choose the same pump
- **Multi-provider support**: DeepSeek, OpenAI, Anthropic, and **Ollama** (local)
- **Dynamic Provider Dispatching** – automatically detects available backends at startup and routes each request intelligently, with built‑in observability for mixed‑provider experiments

## Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`  
   (For Ollama, ensure the Ollama service is running locally – no additional Python package required beyond `requests`).
3. Copy `.env.example` to `.env` and fill in your API keys or enable Ollama (see Configuration below).
4. Run: `python src/main.py`

## Configuration

The system uses a **Dynamic Provider Dispatching** architecture. Instead of hardcoding a single API provider, the simulation runs a lightweight **router** that detects all available LLM backends at startup and dispatches each inference request to one of them dynamically.

This design enables:
- **Zero‑config A/B testing** across different model families (including local Ollama) without manual intervention.
- **Fault tolerance** – if a provider is unreachable, it is automatically excluded from the dispatch pool.
- **Reproducible randomness** – the dispatcher uses a uniform random distribution, but you can seed it for experimental consistency.

### Dynamic Provider Dispatching (How It Works)

1. **Availability Probe**  
   On startup, the dispatcher scans the environment:
   - If `OLLAMA_ENABLED=true` and the Ollama service is responsive, it registers the Ollama endpoint.
   - If `DEEPSEEK_API_KEY` is set (and non‑empty), DeepSeek is registered.
   - If `OPENAI_API_KEY` is set, OpenAI is registered.
   - If `ANTHROPIC_API_KEY` is set, Anthropic is registered.

   *Only providers with valid credentials and a reachable endpoint are added to the active pool.*

2. **Request‑Level Routing**  
   For every individual LLM call (each agent's reasoning step, debate turn, or negotiation message), the dispatcher **picks one provider uniformly at random** from the active pool.  
   This means:
   - If you enable all 4 providers, each request has a 25% chance of hitting any specific one.
   - If you enable only Ollama, 100% of traffic goes local.
   - If you enable OpenAI + Anthropic, traffic is split 50/50.

3. **Observability & Traceability**  
   Every reasoning trace (CoT) stored in the output JSON and HTML report is tagged with:
   - `provider` (e.g., `openai`, `deepseek`, `anthropic`, `ollama`)
   - `model` (e.g., `gpt-4`, `deepseek-reasoner`, `qwen2.5:7b`)  
   This allows you to filter and compare performance, bias, and reasoning patterns across providers post‑experiment.

4. **Forcing a Single Provider**  
   To run a homogeneous experiment (e.g., only Ollama or only GPT‑5), simply **enable only that provider** in your `.env` file. The dispatcher will register a single active provider and route all requests to it.

### Environment Variables

| Variable | Description | Default / Example |
|----------|-------------|-------------------|
| **DeepSeek** | | |
| `DEEPSEEK_API_KEY` | API key for DeepSeek | (your key) |
| `DEEPSEEK_BASE_URL` | Base URL for DeepSeek API | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | Model name for DeepSeek | `deepseek-chat` (or `deepseek-reasoner`) |
| **OpenAI** | | |
| `OPENAI_API_KEY` | API key for OpenAI | (your key) |
| `OPENAI_BASE_URL` | Base URL for OpenAI API | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Model name for OpenAI | `gpt-4` (or `gpt-5.4` if available) |
| **Anthropic** | | |
| `ANTHROPIC_API_KEY` | API key for Anthropic | (your key) |
| `ANTHROPIC_BASE_URL` | Base URL for Anthropic API | `https://api.anthropic.com/v1` |
| `ANTHROPIC_MODEL` | Model name for Anthropic | `claude-3-opus-20240229` (or `claude-opus-5`) |
| **Ollama (Local)** | | |
| `OLLAMA_ENABLED` | Set to `true` to enable Ollama provider | `false` |
| `OLLAMA_BASE_URL` | Ollama server URL (OpenAI‑compatible endpoint) | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Model name for Ollama | `llama2` (or `qwen2.5:7b`, `mistral`) |
| **Runtime** | | |
| `MAX_CONCURRENT` | Max concurrent API calls (per provider pool) | `5` |
| `API_TIMEOUT` | Timeout (seconds) for each request | `600` |
| `SHOW_FULL_COT` | Include full CoT in HTML report | `true` |
| `DISPATCHER_SEED` | (Optional) Seed for random dispatcher to make A/B testing reproducible | (none) |

> **Note**: The legacy `API_PROVIDER` variable is **deprecated** and ignored. The dispatcher automatically handles routing based on the presence of credentials and service availability.

## Output

- `desert_progress_*.json`: full simulation data, including all CoT traces and metadata about which provider was used for each agent.
- `desert_report_*.html`: interactive report with all CoT, decision timelines, and provider labels.

---

Enjoy experimenting with the Monty Hall cognitive bias across diverse LLM families, from cloud APIs to your local Ollama models!