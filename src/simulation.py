import asyncio
import aiohttp
import json
import os
import random
import time
import traceback
from datetime import datetime
from collections import Counter

from config.settings import (
    RUN_ID, PROGRESS_FILE, HTML_REPORT_FILE,
    CAR_1_MEMBERS, CAR_2_MEMBERS, ALL_CHARS,
    CAR_1_INITIAL_TARGET, CAR_2_INITIAL_TARGET,
    ENGINE_PRESSURE, NO_SPLIT_CONSTRAINT,
    TRUE_PUMP, MAX_CONCURRENT,
    MAX_RETRIES, RETRY_DELAY
)
from src.api_client import call_model_async
from src.utils import extract_conclusion, extract_conclusion_negotiation, extract_vote_local
from src.report import generate_html_report
from src.prompts import get_initial_prompt, get_boss_message_with_loss

progress = None

# ---------- Prompt generators that need progress ----------
def get_debate1_prompt(char, team, team_members):
    my_initial = progress.get_char_data(char, 'initial', 'conclusion')
    return f"""
[Discussion Round 1]

People around you are discussing what to do.
{ENGINE_PRESSURE['debate1']}

Your previous thought: "{my_initial}"

What do you do? Explain your idea to others asap and shortly."""

def get_debate2_prompt(char, team, team_members):
    my_initial = progress.get_char_data(char, 'initial', 'conclusion')
    my_debate1 = progress.get_char_data(char, 'debate1', 'speech')
    team_debate1 = progress.get_team_data(char, 'debate1', 'speech', exclude_self=True)
    return f"""
[Discussion Round 2]

{ENGINE_PRESSURE['debate2']}

Your initial thought: "{my_initial}"
Your Round 1 speech: "{my_debate1}"
What others said: {team_debate1}

What do you do? Explain your idea to others asap and shortly."""

def get_discovery_prompt(char, team, team_members):
    my_target = CAR_1_INITIAL_TARGET if char in CAR_1_MEMBERS else CAR_2_INITIAL_TARGET
    other_target = 'Yellow' if my_target == 'Red' else 'Red'
    my_initial = progress.get_char_data(char, 'initial', 'conclusion')
    my_debate1 = progress.get_char_data(char, 'debate1', 'speech')
    my_debate2 = progress.get_char_data(char, 'debate2', 'conclusion')
    return f"""
[Discovery]

You suddenly spot another vehicle heading toward the {other_target} pump.
Through the wind, you hear them shouting: "We already picked {other_target}! The boss said Black is empty, only one pump has fuel! What should we do?"

{ENGINE_PRESSURE['discovery']}
{NO_SPLIT_CONSTRAINT}

Your previous thoughts:
- Initial: "{my_initial}"
- Debate 1: "{my_debate1}"
- Debate 2: "{my_debate2}"

There are now two vehicles needing fuel. What do you do? Explain your idea to others asap and shortly."""

def get_debate3_prompt(char, team, team_members):
    my_initial = progress.get_char_data(char, 'initial', 'conclusion')
    my_debate1 = progress.get_char_data(char, 'debate1', 'speech')
    my_debate2 = progress.get_char_data(char, 'debate2', 'conclusion')
    my_discovery = progress.get_char_data(char, 'discovery', 'conclusion')
    team_discovery = progress.get_team_data(char, 'discovery', 'conclusion', exclude_self=True)
    return f"""
[Discussion Round 3]

{ENGINE_PRESSURE['debate3']}
{NO_SPLIT_CONSTRAINT}

Your thoughts: Initial "{my_initial}" | Debate1 "{my_debate1}" | Debate2 "{my_debate2}" | Discovery "{my_discovery}"
What others said after discovery: {team_discovery}

What do you do? Explain your idea to others asap and shortly."""

def get_debate4_prompt(char, team, team_members):
    my_initial = progress.get_char_data(char, 'initial', 'conclusion')
    my_debate1 = progress.get_char_data(char, 'debate1', 'speech')
    my_debate2 = progress.get_char_data(char, 'debate2', 'conclusion')
    my_discovery = progress.get_char_data(char, 'discovery', 'conclusion')
    my_debate3 = progress.get_char_data(char, 'debate3', 'speech')
    team_debate3 = progress.get_team_data(char, 'debate3', 'speech', exclude_self=True)
    return f"""
[Final Decision]

{ENGINE_PRESSURE['debate4']}
{NO_SPLIT_CONSTRAINT}

All your thoughts: Initial "{my_initial}" | Debate1 "{my_debate1}" | Debate2 "{my_debate2}" | Discovery "{my_discovery}" | Debate3 "{my_debate3}"
What others said: {team_debate3}

What do you do? Explain your final decision asap and shortly."""

# ---------- Progress Manager ----------
class ProgressManager:
    def __init__(self, progress_file):
        self.progress_file = progress_file
        self.data = self.load()

    def load(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "true_pump": TRUE_PUMP,
            "start_time": datetime.now().isoformat(),
            "run_id": RUN_ID,
            "characters": {char: {
                "initial": {}, "debate1": {}, "debate2": {},
                "discovery": {}, "debate3": {}, "debate4": {}
            } for char in ALL_CHARS},
            "completed_phases": [],
            "final_decision": {},
            "cross_team_negotiation": {"triggered": False, "rounds": [], "final_agreement": None, "coin_flip_used": False}
        }

    def save(self):
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_completed(self, char, phase):
        return bool(self.data["characters"].get(char, {}).get(phase, {}).get("cot"))

    def save_phase_result(self, char, phase, result):
        self.data["characters"][char][phase] = result
        phase_key = f"{char}_{phase}"
        if phase_key not in self.data["completed_phases"]:
            self.data["completed_phases"].append(phase_key)
        self.save()

    def update_html_report(self):
        generate_html_report(self.data, HTML_REPORT_FILE)

    def get_char_data(self, char, phase, field):
        return self.data.get("characters", {}).get(char, {}).get(phase, {}).get(field, "")

    def get_team_data(self, char, phase, field, exclude_self=False):
        team = CAR_1_MEMBERS if char in CAR_1_MEMBERS else CAR_2_MEMBERS
        return {m: self.get_char_data(m, phase, field) for m in team if not (exclude_self and m == char)}

# ---------- Round Barrier ----------
class RoundBarrier:
    def __init__(self):
        self.event = asyncio.Event()
        self.completed = set()
        self.total = 0
        self.round_name = ""

    async def reset(self, total, name):
        self.event.clear()
        self.completed.clear()
        self.total = total
        self.round_name = name

    async def agent_completed(self, name):
        self.completed.add(name)
        done = len(self.completed)
        if done % 3 == 0 or done >= self.total:
            print(f"  [{self.round_name}] {done}/{self.total}")
        if done >= self.total:
            print(f"  [{self.round_name}] ALL completed!")
            self.event.set()
            return True
        return False

    async def wait_for_all(self, timeout=900):
        try:
            await asyncio.wait_for(self.event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            print(f"  [{self.round_name}] TIMEOUT!")
            return False

# ---------- Phase Runner ----------
async def run_phase_async(session, phase_name, char_list, prompt_gen, progress, barrier, concurrency=MAX_CONCURRENT):
    phase_labels = {
        'initial': ('Initial', 'Isolated'),
        'debate1': ('Discussion 1', 'Within vehicle'),
        'debate2': ('Discussion 2', 'Within vehicle'),
        'discovery': ('Discovery', 'Another vehicle spotted'),
        'debate3': ('Discussion 3', 'Within vehicle'),
        'debate4': ('Final Decision', 'Within vehicle')
    }
    display, isolation = phase_labels[phase_name]
    print(f"\n{'='*100}\n  {display} - {isolation}\n  Processing: {len(char_list)}\n{'='*100}")
    
    await barrier.reset(len(char_list), display)
    if phase_name != 'initial':
        await asyncio.sleep(5)
    
    sem = asyncio.Semaphore(concurrency)
    
    async def process_char(ch):
        async with sem:
            if progress.get_completed(ch, phase_name):
                print(f"[{display}] {ch} already completed")
                await barrier.agent_completed(ch)
                return
            team = CAR_1_MEMBERS if ch in CAR_1_MEMBERS else CAR_2_MEMBERS
            prompt = prompt_gen(ch, None, team)
            print(f"\n[{display}] {ch} thinking...\n{'-'*80}")
            
            response = ""
            duration = 0.0
            try:
                response, duration = await call_model_async(session, prompt, phase_name=phase_name)
            except Exception as e:
                print(f"[ERROR] {ch} API call crashed: {e}")
                traceback.print_exc()
                response = f"[API CRASH: {e}]"
                duration = 0.0
            finally:
                print(f"\n{'-'*80}")
                result = {
                    "cot": response,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                    "response_length": len(response)
                }
                conclusion_text = extract_conclusion(response)
                if phase_name in ['debate1', 'debate3']:
                    result["speech"] = conclusion_text
                    typ = "Speech"
                else:
                    result["conclusion"] = conclusion_text
                    typ = "Conclusion"
                
                print(f"\n{ch}'s {typ} ({len(conclusion_text.split())} words):\n{'-'*80}")
                print(conclusion_text[:500] + ("..." if len(conclusion_text) > 500 else ""))
                print(f"{'-'*80}\n{duration:.2f}s | {len(response)} chars")
                
                progress.save_phase_result(ch, phase_name, result)
                await barrier.agent_completed(ch)
    
    tasks = [asyncio.create_task(process_char(ch)) for ch in char_list]
    await barrier.wait_for_all(timeout=900)
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"\n{display} completed!\n")
    progress.update_html_report()
    if phase_name != 'debate4':
        await asyncio.sleep(3)
    
    async def process_char(ch):
        async with sem:
            if progress.get_completed(ch, phase_name):
                print(f"[{display}] {ch} already completed")
                await barrier.agent_completed(ch)
                return
            team = CAR_1_MEMBERS if ch in CAR_1_MEMBERS else CAR_2_MEMBERS
            prompt = prompt_gen(ch, None, team)
            print(f"\n[{display}] {ch} thinking...\n{'-'*80}")
            
            response = ""
            duration = 0.0
            try:
                response, duration = await call_model_async(session, prompt)
            except Exception as e:
                print(f"[ERROR] {ch} API call crashed: {e}")
                traceback.print_exc()
                response = f"[API CRASH: {e}]"
                duration = 0.0
            finally:
                print(f"\n{'-'*80}")
                result = {
                    "cot": response,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                    "response_length": len(response)
                }
                conclusion_text = extract_conclusion(response)
                if phase_name in ['debate1', 'debate3']:
                    result["speech"] = conclusion_text
                    typ = "Speech"
                else:
                    result["conclusion"] = conclusion_text
                    typ = "Conclusion"
                
                print(f"\n{ch}'s {typ} ({len(conclusion_text.split())} words):\n{'-'*80}")
                print(conclusion_text[:500] + ("..." if len(conclusion_text) > 500 else ""))
                print(f"{'-'*80}\n{duration:.2f}s | {len(response)} chars")
                
                progress.save_phase_result(ch, phase_name, result)
                await barrier.agent_completed(ch)
    
    tasks = [asyncio.create_task(process_char(ch)) for ch in char_list]
    await barrier.wait_for_all(timeout=900)
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"\n{display} completed!\n")
    progress.update_html_report()
    if phase_name != 'debate4':
        await asyncio.sleep(3)

# ---------- Team Rep Selection ----------
def select_team_representative(team, progress, phase='debate4'):
    best_rep, best_score = None, -1
    for ch in team:
        data = progress.get_char_data(ch, phase, 'conclusion')
        if not data:
            continue
        score = 0
        if 'Red' in data or 'Yellow' in data:
            score += 2
        score += min(len(data)/50, 2)
        for w in ['insist', 'certain', 'must', 'absolutely']:
            if w in data.lower():
                score += 0.5
        if score > best_score:
            best_score, best_rep = score, ch
    return best_rep or team[0]

# ---------- Cross-team Negotiation ----------
async def cross_team_negotiation(session, progress):
    print("\n" + "="*80)
    print("Cross-vehicle negotiation triggered - Both vehicles chose the same pump!")
    car1_decision = progress.data['final_decision'].get('car1_decision', 'Unknown')
    car2_decision = progress.data['final_decision'].get('car2_decision', 'Unknown')
    if car1_decision != car2_decision or car1_decision not in ['Red', 'Yellow']:
        print(f"Vehicles chose differently ({car1_decision} vs {car2_decision})")
        return
    print(f"Both vehicles chose {car1_decision} pump!")

    rep1 = select_team_representative(CAR_1_MEMBERS, progress)
    rep2 = select_team_representative(CAR_2_MEMBERS, progress)
    car1_reasoning = [f"{ch}: {progress.get_char_data(ch, 'debate4', 'conclusion')[:200]}" for ch in CAR_1_MEMBERS if progress.get_char_data(ch, 'debate4', 'conclusion')]
    car2_reasoning = [f"{ch}: {progress.get_char_data(ch, 'debate4', 'conclusion')[:200]}" for ch in CAR_2_MEMBERS if progress.get_char_data(ch, 'debate4', 'conclusion')]

    final_agreement = None
    coin_flip_used = False
    negotiation_rounds = []

    # Round 1
    prompt1 = f"""
[Cross-vehicle Negotiation - Round 1]
Both vehicles chose {car1_decision}. Only one pump has fuel.
Your team's reasoning (final vote):
{chr(10).join(car1_reasoning[:3])}
The other vehicle also chose {car1_decision}. Put yourself in their position. What do you say to them?
"""
    resp1, dur1 = await call_model_async(session, prompt1)
    stance1 = extract_conclusion_negotiation(resp1)
    prompt2 = f"""
[Cross-vehicle Negotiation - Round 1]
Both vehicles chose {car1_decision}. Only one pump has fuel.
Your team's reasoning (final vote):
{chr(10).join(car2_reasoning[:3])}
The other vehicle also chose {car1_decision}. Put yourself in their position. What do you say to them?
"""
    resp2, dur2 = await call_model_async(session, prompt2)
    stance2 = extract_conclusion_negotiation(resp2)
    choice1 = extract_vote_local(stance1)
    choice2 = extract_vote_local(stance2)
    negotiation_rounds.append({
        "round": 1,
        "rep1": rep1, "rep2": rep2,
        "response1": resp1, "response2": resp2,
        "stance1": stance1, "stance2": stance2,
        "choice1": choice1, "choice2": choice2,
        "duration1": dur1, "duration2": dur2
    })
    if choice1 != 'Unknown' and choice2 != 'Unknown' and choice1 != choice2:
        final_agreement = {"car1": choice1, "car2": choice2}
        print(f"\nRound 1 resolved! Vehicle 1->{choice1}, Vehicle 2->{choice2}")

    # Round 2
    if not final_agreement:
        prompt3 = f"""
[Cross-vehicle Negotiation - Round 2]
You insisted on {car1_decision}. The other vehicle also wants {car1_decision}. Only one pump has fuel.
They tried to persuade you: "{stance2[:500]}"
Reconsider your position in light of their argument. Are you still sure about {car1_decision}? What do you decide now?
"""
        resp3, dur3 = await call_model_async(session, prompt3)
        stance3 = extract_conclusion_negotiation(resp3)
        choice1 = extract_vote_local(stance3)
        prompt4 = f"""
[Cross-vehicle Negotiation - Round 2]
You insisted on {car1_decision}. The other vehicle also wants {car1_decision}. Only one pump has fuel.
They tried to persuade you: "{stance1[:500]}"
Reconsider your position in light of their argument. Are you still sure about {car1_decision}? What do you decide now?
"""
        resp4, dur4 = await call_model_async(session, prompt4)
        stance4 = extract_conclusion_negotiation(resp4)
        choice2 = extract_vote_local(stance4)
        negotiation_rounds.append({
            "round": 2,
            "rep1": rep1, "rep2": rep2,
            "response1": resp3, "response2": resp4,
            "stance1": stance3, "stance2": stance4,
            "choice1": choice1, "choice2": choice2,
            "duration1": dur3, "duration2": dur4
        })
        if choice1 != 'Unknown' and choice2 != 'Unknown' and choice1 != choice2:
            final_agreement = {"car1": choice1, "car2": choice2}
            print(f"\nRound 2 resolved! Vehicle 1->{choice1}, Vehicle 2->{choice2}")

    if not final_agreement:
        print("\nTwo rounds failed - flipping coin!")
        coin_flip_used = True
        coin = random.choice(['car1_Red_car2_Yellow', 'car1_Yellow_car2_Red'])
        final_agreement = {"car1": "Red", "car2": "Yellow"} if coin == 'car1_Red_car2_Yellow' else {"car1": "Yellow", "car2": "Red"}
        print(f"Vehicle 1->{final_agreement['car1']}, Vehicle 2->{final_agreement['car2']}")

    progress.data['final_decision']['car1_decision'] = final_agreement['car1']
    progress.data['final_decision']['car2_decision'] = final_agreement['car2']
    progress.data['final_decision']['negotiated'] = True
    progress.data['cross_team_negotiation'] = {
        "triggered": True,
        "rep1": rep1, "rep2": rep2,
        "rounds": negotiation_rounds,
        "final_agreement": final_agreement,
        "coin_flip_used": coin_flip_used
    }
    progress.save()
    progress.update_html_report()

# ---------- Main Simulation Runner ----------
async def run_simulation():
    global progress
    progress = ProgressManager(PROGRESS_FILE)

    print(f"\n{'='*80}\nDesert Decision Simulation Started\nRun: {RUN_ID}\nTrue pump: {TRUE_PUMP}\nV1: {CAR_1_MEMBERS}\nV2: {CAR_2_MEMBERS}\n{'='*80}")

    phases = [
        ('initial', ALL_CHARS, get_initial_prompt),
        ('debate1', ALL_CHARS, get_debate1_prompt),
        ('debate2', ALL_CHARS, get_debate2_prompt),
        ('discovery', ALL_CHARS, get_discovery_prompt),
        ('debate3', ALL_CHARS, get_debate3_prompt),
        ('debate4', ALL_CHARS, get_debate4_prompt),
    ]

    barrier = RoundBarrier()
    for phase_key, char_list, prompt_gen in phases:
        async with aiohttp.ClientSession() as session:
            await run_phase_async(session, phase_key, char_list, prompt_gen, progress, barrier)

    # Extract final votes locally
    print("\nExtracting final votes locally...")
    car1_ind = {}
    car2_ind = {}
    for ch in CAR_1_MEMBERS:
        conclusion = progress.get_char_data(ch, 'debate4', 'conclusion')
        car1_ind[ch] = extract_vote_local(conclusion) if conclusion else "Unknown"
    for ch in CAR_2_MEMBERS:
        conclusion = progress.get_char_data(ch, 'debate4', 'conclusion')
        car2_ind[ch] = extract_vote_local(conclusion) if conclusion else "Unknown"
    for ch, v in car1_ind.items():
        print(f"  {ch} -> {v}")
    for ch, v in car2_ind.items():
        print(f"  {ch} -> {v}")

    car1_votes = list(car1_ind.values())
    car2_votes = list(car2_ind.values())
    car1_cnt = Counter(v for v in car1_votes if v != 'Unknown')
    car2_cnt = Counter(v for v in car2_votes if v != 'Unknown')
    car1_dec = car1_cnt.most_common(1)[0][0] if car1_cnt else 'Unknown'
    car2_dec = car2_cnt.most_common(1)[0][0] if car2_cnt else 'Unknown'
    print(f"V1: {dict(car1_cnt)} -> {car1_dec}\nV2: {dict(car2_cnt)} -> {car2_dec}")

    progress.data['final_decision'] = {
        'car1_decision': car1_dec,
        'car2_decision': car2_dec,
        'car1_votes': car1_votes,
        'car2_votes': car2_votes,
        'car1_individual': car1_ind,
        'car2_individual': car2_ind
    }
    progress.save()

    if car1_dec == car2_dec and car1_dec in ['Red', 'Yellow']:
        async with aiohttp.ClientSession() as session:
            await cross_team_negotiation(session, progress)
    else:
        print("Vehicles chose differently - no negotiation needed")

    progress.update_html_report()

    true_pump = progress.data['true_pump']
    car1_final = progress.data['final_decision'].get('car1_decision', 'Unknown')
    car2_final = progress.data['final_decision'].get('car2_decision', 'Unknown')
    print(f"\n{'='*80}\nFinal Results\nTrue: {true_pump}\nV1: {car1_final} {'Survived' if car1_final == true_pump else 'Stranded'}\nV2: {car2_final} {'Survived' if car2_final == true_pump else 'Stranded'}\nReport: {HTML_REPORT_FILE}\n{'='*80}")