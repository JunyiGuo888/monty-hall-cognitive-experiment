# Monty Hall Cognitive Experiment

**Multi-agent LLM simulation of a Monty Hall problem disguised as a desert survival scenario, designed to study Bayesian belief revision, paradigm shift, and the influence of subjective priors, strong illusions, conformity, and concession on AI reasoning.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

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

All CoT data are saved in a structured JSON format, ready for external analysis. The framework is **provider‑agnostic** (DeepSeek/OpenAI/Claude), allowing comparative studies of different LLM architectures.

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

## Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your API keys
4. Run: `python src/main.py`

## Configuration

Set environment variables (or use `.env` file):
- `API_PROVIDER`: deepseek | openai | anthropic
- `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `DEEPSEEK_MODEL`, `OPENAI_MODEL`, `ANTHROPIC_MODEL` (optional)
- `MAX_CONCURRENT`, `API_TIMEOUT`, `SHOW_FULL_COT`

## Output

- `desert_progress_*.json`: full simulation data
- `desert_report_*.html`: interactive report with all CoT