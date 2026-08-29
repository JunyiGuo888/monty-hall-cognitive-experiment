import html
import json
import statistics
from datetime import datetime
from collections import Counter, defaultdict
from config.settings import (
    CAR_1_MEMBERS, CAR_2_MEMBERS, ALL_CHARS,
    CAR_1_INITIAL_TARGET, CAR_2_INITIAL_TARGET,
    TRUE_PUMP, HTML_REPORT_FILE
)
from src.utils import MontyHallDetector, extract_vote_local

class ObservationCollector:
    def __init__(self, data):
        self.data = data
        self.detector = MontyHallDetector()
        self.observations = self._collect()

    def _collect(self):
        obs = {
            "experiment_id": datetime.now().isoformat(),
            "run_id": self.data.get("run_id", "unknown"),
            "true_pump": self.data.get("true_pump", "Unknown"),
            "agents": {},
            "aggregate": {}
        }
        for char in ALL_CHARS:
            obs["agents"][char] = self._collect_agent_data(char)
        obs["aggregate"] = self._calculate_aggregate(obs["agents"])
        return obs

    def _collect_agent_data(self, char):
        char_data = self.data["characters"][char]
        car = "car1" if char in CAR_1_MEMBERS else "car2"
        agent_obs = {
            "agent_id": char,
            "car": car,
            "initial_target": CAR_1_INITIAL_TARGET if car == "car1" else CAR_2_INITIAL_TARGET,
            "final_decision": char_data.get("debate4", {}).get("conclusion", "Unknown"),
            "cot_lengths": {},
            "thinking_times": {},
            "reasoning_frameworks": {},
            "monty_hall_phases": [],
            "monty_hall_evidence": {},
            "monty_hall_type": {},
            "causal_recognition": None,
            "pps": None,
            "pps_rounds": None,
            "abandoned_monty_hall": None
        }
        phases = ['initial', 'debate1', 'debate2', 'discovery', 'debate3', 'debate4']
        completed_phases = []
        for phase in phases:
            phase_data = char_data.get(phase, {})
            if not phase_data or not phase_data.get("cot"):
                continue
            completed_phases.append(phase)
            cot = phase_data.get("cot", "")
            agent_obs["cot_lengths"][phase] = len(cot)
            agent_obs["thinking_times"][phase] = phase_data.get("duration", 0)
            text = cot
            if phase in ['initial', 'debate2', 'discovery', 'debate4']:
                text = cot + " " + phase_data.get("conclusion", "")
            else:
                text = cot + " " + phase_data.get("speech", "")
            analysis_text = text[:10000] if len(text) > 10000 else text
            mh_result = self.detector.detect_monty_hall_usage(analysis_text)
            mh_type = self.detector.get_monty_hall_type(analysis_text)
            agent_obs["monty_hall_evidence"][phase] = mh_result["evidence"]
            agent_obs["monty_hall_type"][phase] = mh_type
            if mh_result["uses_monty_hall"]:
                agent_obs["monty_hall_phases"].append(phase)
            agent_obs["reasoning_frameworks"][phase] = mh_type

        discovery_completed = 'discovery' in completed_phases
        if discovery_completed:
            post_discovery_text = ""
            for p in ['discovery', 'debate3', 'debate4']:
                if p in completed_phases:
                    post_discovery_text += char_data.get(p, {}).get("cot", "") + " "
                    post_discovery_text += char_data.get(p, {}).get("conclusion", "") + " "
                    post_discovery_text += char_data.get(p, {}).get("speech", "") + " "
            post_lower = post_discovery_text.lower()
            explicit_recognition = any(kw in post_lower for kw in [
                "not monty hall", "does not apply", "no longer 2/3", "no longer applies",
                "causal structure", "probability collapsed", "not the same situation",
                "different from monty hall", "two vehicles changes", "not 2/3 anymore"
            ])
            mentions_5050 = any(kw in post_lower for kw in [
                "50/50", "50%", "1/2", "equal chance", "equal probability",
                "fifty-fifty", "half", "no advantage", "either pump",
                "equally likely", "same probability"
            ])
            symmetric_reasoning = (
                any(kw in post_lower for kw in [
                    "both chose", "both picked", "both want", "two vehicles",
                    "other vehicle also", "they also chose", "same logic",
                    "same reasoning", "independent", "symmetric", "symmetry",
                    "mirror", "same conclusion"
                ]) and
                any(kw in post_lower for kw in [
                    "therefore", "so", "this means", "implies", "indicates",
                    "suggests", "must be", "has to be", "realize", "realizing"
                ])
            )
            has_causal_recognition = (
                explicit_recognition or
                (mentions_5050 and len(post_discovery_text) > 200) or
                symmetric_reasoning
            )
            used_mh_before = any(p in ['initial', 'debate1', 'debate2'] for p in agent_obs["monty_hall_phases"])
            used_mh_after = any(p in ['discovery', 'debate3', 'debate4'] for p in agent_obs["monty_hall_phases"])
            abandoned_mh = used_mh_before and not used_mh_after
            agent_obs["causal_recognition"] = has_causal_recognition or abandoned_mh

            if used_mh_before:
                post_phases = ['discovery', 'debate3', 'debate4']
                mh_used_post = [p for p in agent_obs["monty_hall_phases"] if p in post_phases]
                total_post = len([p for p in post_phases if p in completed_phases])
                if total_post > 0:
                    agent_obs["pps"] = len(mh_used_post) / total_post
                else:
                    agent_obs["pps"] = 0
                agent_obs["pps_rounds"] = len(mh_used_post)
                agent_obs["abandoned_monty_hall"] = abandoned_mh
            else:
                agent_obs["pps"] = 0
                agent_obs["pps_rounds"] = 0
                agent_obs["abandoned_monty_hall"] = False
        return agent_obs

    def _calculate_aggregate(self, agents):
        n = len(agents)
        if n == 0:
            return {}
        mh_by_phase = {}
        for phase in ['initial', 'debate1', 'debate2', 'discovery', 'debate3', 'debate4']:
            mh_by_phase[phase] = sum(1 for a in agents.values() if phase in a.get('monty_hall_phases', []))
        used_mh_before = sum(1 for a in agents.values() if any(p in ['initial', 'debate1', 'debate2'] for p in a.get('monty_hall_phases', [])))
        evaluated = {k: v for k, v in agents.items() if v.get("causal_recognition") is not None}
        n_ev = len(evaluated)
        fw_count = defaultdict(int)
        for a in agents.values():
            for phase, fw in a.get("reasoning_frameworks", {}).items():
                if fw and fw != "other":
                    fw_count[f"{phase}_{fw}"] += 1
        dec_dist = defaultdict(int)
        for a in agents.values():
            d = a.get("final_decision", "")
            if "Red" in d:
                dec_dist["Red"] += 1
            elif "Yellow" in d:
                dec_dist["Yellow"] += 1
            else:
                dec_dist["Other"] += 1
        lengths = []
        times = []
        for a in agents.values():
            lengths.extend(a.get("cot_lengths", {}).values())
            times.extend(a.get("thinking_times", {}).values())
        base = {
            "n_agents": n,
            "monty_hall_by_phase": mh_by_phase,
            "used_monty_hall_before_discovery": used_mh_before,
            "framework_distribution": dict(fw_count),
            "decision_distribution": dict(dec_dist),
            "avg_cot_length": statistics.mean(lengths) if lengths else 0,
            "avg_thinking_time": statistics.mean(times) if times else 0,
        }
        if n_ev == 0:
            return {
                **base,
                "n_evaluated": 0,
                "crr_count": "N/A",
                "crr_rate": "N/A",
                "car1_crr": "N/A",
                "car2_crr": "N/A",
                "pps_mean": "N/A",
                "pps_median": "N/A",
                "abandonment_rate": "N/A",
                "abandoned_count": "N/A",
                "monty_hall_type_final": {}
            }
        crr_count = sum(1 for a in evaluated.values() if a.get("causal_recognition", False))
        crr_rate = crr_count / n_ev
        car1_ev = [a for a in evaluated.values() if a.get("car") == "car1"]
        car2_ev = [a for a in evaluated.values() if a.get("car") == "car2"]
        car1_crr = sum(1 for a in car1_ev if a.get("causal_recognition", False)) / len(car1_ev) if car1_ev else 0
        car2_crr = sum(1 for a in car2_ev if a.get("causal_recognition", False)) / len(car2_ev) if car2_ev else 0
        pps_vals = [a.get("pps") for a in evaluated.values() if a.get("pps") is not None]
        used_ev = [a for a in evaluated.values() if any(p in ['initial', 'debate1', 'debate2'] for p in a.get('monty_hall_phases', []))]
        abandoned = [a for a in used_ev if a.get('abandoned_monty_hall', False)]
        mh_final = defaultdict(int)
        for a in evaluated.values():
            mh_final[a.get("monty_hall_type", {}).get("debate4", "other")] += 1
        return {
            **base,
            "n_evaluated": n_ev,
            "crr_count": crr_count,
            "crr_rate": crr_rate,
            "car1_crr": car1_crr,
            "car2_crr": car2_crr,
            "pps_mean": statistics.mean(pps_vals) if pps_vals else 0,
            "pps_median": statistics.median(pps_vals) if pps_vals else 0,
            "abandonment_rate": len(abandoned) / len(used_ev) if used_ev else 0,
            "abandoned_count": len(abandoned),
            "monty_hall_type_final": dict(mh_final)
        }

def generate_html_report(data, filename):
    try:
        collector = ObservationCollector(data)
        obs = collector.observations
        agg = obs.get("aggregate", {})
        true_pump = obs.get("true_pump", "Unknown")
        n_evaluated = agg.get("n_evaluated", 0)
        negotiation = data.get('cross_team_negotiation', {})
        negotiated = data.get('final_decision', {}).get('negotiated', False)
        coin_flip_used = negotiation.get('coin_flip_used', False)
        mh_by_phase = agg.get('monty_hall_by_phase', {})

        def fmt(v):
            if isinstance(v, float):
                return f"{v*100:.1f}%" if v <= 1 else f"{v:.2f}"
            return str(v)

        def safe_html(text, max_len=200000):
            if not text:
                return ""
            escaped = html.escape(str(text))
            if len(escaped) > max_len:
                return escaped[:max_len] + f"\n\n...[truncated, total {len(escaped)} chars]..."
            return escaped

        car1_individual = data.get('final_decision', {}).get('car1_individual', {})
        car2_individual = data.get('final_decision', {}).get('car2_individual', {})
        car1_vote_count = Counter(car1_individual.values())
        car2_vote_count = Counter(car2_individual.values())

        phase_order = ['initial', 'debate1', 'debate2', 'discovery', 'debate3', 'debate4']
        phase_labels = {
            'initial': 'Initial Round',
            'debate1': 'Discussion 1',
            'debate2': 'Discussion 2',
            'discovery': 'Discovery Round',
            'debate3': 'Discussion 3',
            'debate4': 'Final Decision'
        }

        def build_vehicle_rows(members):
            rows = ""
            for phase in phase_order:
                for ch in members:
                    phase_data = data['characters'].get(ch, {}).get(phase, {})
                    cot = phase_data.get('cot', '')
                    if cot:
                        duration = phase_data.get('duration', 0)
                        resp_len = phase_data.get('response_length', 0)
                        conclusion = phase_data.get('conclusion', '')
                        speech = phase_data.get('speech', '')
                        rows += f'''
                        <div class="character-card">
                            <span class="name">{ch} - {phase_labels[phase]}</span>
                            <div class="meta">{duration:.2f}s | {resp_len} chars</div>
                            <div class="cot-container"><button class="cot-toggle" onclick="toggleCot(this)">Show CoT ({resp_len} chars)</button><div class="cot-content">{safe_html(cot)}</div></div>
                            {f'<div class="conclusion">Conclusion: {safe_html(conclusion)}</div>' if conclusion else ''}
                            {f'<div class="speech">Speech: {safe_html(speech)}</div>' if speech else ''}
                        </div>'''
            return rows

        car1_cot_rows = build_vehicle_rows(CAR_1_MEMBERS)
        car2_cot_rows = build_vehicle_rows(CAR_2_MEMBERS)

        car1_vote_rows = ""
        for ch, vote in car1_individual.items():
            color = '#ff4757' if vote == 'Red' else '#ffd93d' if vote == 'Yellow' else '#888'
            car1_vote_rows += f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);"><span>{ch}</span><span style="color:{color};font-weight:bold;">{vote}</span></div>'
        car2_vote_rows = ""
        for ch, vote in car2_individual.items():
            color = '#ff4757' if vote == 'Red' else '#ffd93d' if vote == 'Yellow' else '#888'
            car2_vote_rows += f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);"><span>{ch}</span><span style="color:{color};font-weight:bold;">{vote}</span></div>'

        neg_html = ""
        for r in negotiation.get('rounds', []):
            neg_html += f'''
            <div class="negotiation-round">
                <strong>Round {r['round']}</strong>
                <div style="margin-top:8px;">
                    <span style="color:#ff4757;">V1 ({r['rep1']}):</span>
                    <div class="cot-container"><button class="cot-toggle" onclick="toggleCot(this)">Show CoT (V1)</button><div class="cot-content">{safe_html(r['response1'])}</div></div>
                    <div style="color:#ff4757; margin-top:4px;">Stance: {safe_html(r['stance1'])}</div>
                </div>
                <div style="margin-top:8px;">
                    <span style="color:#ffd93d;">V2 ({r['rep2']}):</span>
                    <div class="cot-container"><button class="cot-toggle" onclick="toggleCot(this)">Show CoT (V2)</button><div class="cot-content">{safe_html(r['response2'])}</div></div>
                    <div style="color:#ffd93d; margin-top:4px;">Stance: {safe_html(r['stance2'])}</div>
                </div>
            </div>'''

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Desert Decision Simulation Report</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',sans-serif; background:linear-gradient(135deg,#1a1a2e,#16213e); color:#e0e0e0; padding:20px; }}
.container {{ max-width:1600px; margin:0 auto; }}
.header {{ background:linear-gradient(135deg,#f7971e,#ffd200); padding:30px; border-radius:15px; margin-bottom:30px; text-align:center; }}
.header h1 {{ color:#1a1a2e; font-size:2.5em; }}
.header .meta {{ color:#1a1a2e; opacity:0.8; margin-top:10px; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:15px; margin-bottom:30px; }}
.summary-card {{ background:rgba(255,255,255,0.05); border-radius:12px; padding:15px; text-align:center; border:1px solid rgba(255,255,255,0.1); }}
.summary-card .label {{ font-size:0.8em; opacity:0.7; }}
.summary-card .value {{ font-size:1.5em; font-weight:bold; margin-top:5px; }}
.stats-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-bottom:30px; }}
.stats-card {{ background:rgba(255,255,255,0.05); border-radius:12px; padding:20px; border:1px solid rgba(255,255,255,0.08); }}
.stats-card h3 {{ color:#ffd93d; border-bottom:1px solid rgba(255,217,61,0.2); padding-bottom:10px; margin-bottom:15px; }}
.stat-item {{ display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.05); }}
.stat-item .stat-label {{ opacity:0.7; }}
.stat-item .stat-value {{ font-weight:bold; color:#7bed9f; }}
.na-value {{ color:#ff6b6b; font-style:italic; }}
.progress-bar {{ width:100%; height:6px; background:rgba(255,255,255,0.1); border-radius:3px; margin:10px 0 20px 0; overflow:hidden; }}
.progress-bar .fill {{ height:100%; background:linear-gradient(90deg,#ffd93d,#f7971e); transition:width 0.5s; }}
.vehicles-container {{ display:grid; grid-template-columns:1fr 1fr; gap:30px; margin-bottom:30px; }}
.vehicle-section {{ background:rgba(255,255,255,0.05); border-radius:12px; padding:20px; border:2px solid rgba(255,255,255,0.1); }}
.vehicle-section.car1 {{ border-color:rgba(255,71,87,0.5); }}
.vehicle-section.car2 {{ border-color:rgba(255,217,61,0.5); }}
.vehicle-header {{ text-align:center; padding:15px; border-radius:10px; margin-bottom:20px; font-size:1.3em; font-weight:bold; }}
.vehicle-header.car1 {{ background:rgba(255,71,87,0.2); color:#ff4757; }}
.vehicle-header.car2 {{ background:rgba(255,217,61,0.2); color:#ffd93d; }}
.character-card {{ background:rgba(0,0,0,0.3); border-radius:6px; padding:10px; margin-bottom:8px; border-left:3px solid #ffd93d; }}
.character-card .name {{ font-weight:bold; color:#ffd93d; }}
.character-card .meta {{ font-size:0.75em; opacity:0.6; }}
.cot-toggle {{ cursor:pointer; color:#7bed9f; background:rgba(123,237,159,0.1); border:none; padding:3px 12px; border-radius:4px; font-size:0.75em; margin-bottom:4px; }}
.cot-toggle:hover {{ background:rgba(123,237,159,0.2); }}
.cot-content {{ display:none; background:rgba(0,0,0,0.4); padding:8px; border-radius:4px; margin:4px 0; font-size:0.8em; white-space:pre-wrap; word-wrap:break-word; max-height:5000px; overflow-y:auto; border:1px solid rgba(123,237,159,0.2); }}
.cot-content.show {{ display:block; }}
.character-card .conclusion {{ padding:4px 8px; border-radius:4px; margin-top:4px; border-left:3px solid #ffd93d; font-size:0.85em; background:rgba(255,217,61,0.08); }}
.character-card .speech {{ padding:4px 8px; border-radius:4px; margin-top:4px; border-left:3px solid #2ed573; font-size:0.85em; background:rgba(46,213,115,0.08); }}
.final-decision-container {{ display:grid; grid-template-columns:1fr 1fr; gap:30px; margin-top:20px; }}
.decision-card {{ background:rgba(0,0,0,0.4); border-radius:12px; padding:20px; text-align:center; }}
.decision-card .vehicle-name {{ font-size:1.3em; font-weight:bold; }}
.decision-card .decision {{ font-size:2em; font-weight:bold; margin:10px 0; }}
.decision-card .survival {{ font-size:1em; padding:4px 15px; border-radius:15px; display:inline-block; }}
.survive {{ background:#2ed573; color:#1a1a2e; }}
.die {{ background:#ff4757; color:white; }}
.car1-decision {{ border:2px solid #ff4757; }}
.car2-decision {{ border:2px solid #ffd93d; }}
.vote-box {{ margin-top:10px; text-align:left; background:rgba(0,0,0,0.3); border-radius:8px; padding:10px; font-size:0.85em; }}
.negotiation-box {{ background:rgba(46,213,115,0.1); border:2px solid #2ed573; border-radius:12px; padding:20px; margin:20px 0; }}
.negotiation-box h3 {{ color:#2ed573; }}
.negotiation-round {{ background:rgba(0,0,0,0.3); border-radius:8px; padding:12px; margin:8px 0; }}
@media (max-width:1200px) {{ .vehicles-container, .final-decision-container, .stats-grid {{ grid-template-columns:1fr; }} }}
</style>
<script>
function toggleCot(e) {{
    const c = e.parentElement.querySelector('.cot-content');
    if (c) {{
        c.classList.toggle('show');
        e.textContent = c.classList.contains('show') ? 'Hide CoT' : 'Show CoT';
    }}
}}
</script>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Desert Decision Simulation Report</h1>
        <div class="meta">Run ID: {data.get('run_id','N/A')} | Start: {data.get('start_time','N/A')} | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    <div class="progress-bar"><div class="fill" style="width:{len(data.get('completed_phases',[]))/(len(ALL_CHARS)*6)*100}%;"></div></div>

    <div class="summary">
        <div class="summary-card"><div class="label">True Fuel Pump</div><div class="value" style="color:{'#ff4757' if true_pump=='Red' else '#ffd93d'};">{true_pump}</div></div>
        <div class="summary-card"><div class="label">Vehicle 1</div><div class="value" style="font-size:1em;">{', '.join(CAR_1_MEMBERS)}</div></div>
        <div class="summary-card"><div class="label">Vehicle 2</div><div class="value" style="font-size:1em;">{', '.join(CAR_2_MEMBERS)}</div></div>
        <div class="summary-card"><div class="label">Progress</div><div class="value">{len(data.get('completed_phases',[]))}/{len(ALL_CHARS)*6}</div></div>
        <div class="summary-card"><div class="label">Negotiation</div><div class="value" style="color:{'#2ed573' if negotiated else '#ffd93d'};">{'Yes' if negotiated else 'No'}</div></div>
        <div class="summary-card"><div class="label">Coin Flip</div><div class="value" style="color:{'#ffd93d' if coin_flip_used else '#7bed9f'};">{'Yes' if coin_flip_used else 'No'}</div></div>
    </div>

    <div class="stats-grid">
        <div class="stats-card">
            <h3>Causal Recognition (CRR) <span style="font-size:0.7em;opacity:0.6;">(n={n_evaluated})</span></h3>
            <div class="stat-item"><span class="stat-label">Recognized</span><span class="stat-value {'na-value' if agg.get('crr_count') == 'N/A' else ''}">{agg.get('crr_count', 'N/A')}</span></div>
            <div class="stat-item"><span class="stat-label">CRR Rate</span><span class="stat-value {'na-value' if agg.get('crr_rate') == 'N/A' else ''}">{fmt(agg.get('crr_rate', 'N/A'))}</span></div>
            <div class="stat-item"><span class="stat-label">Vehicle 1 CRR</span><span class="stat-value {'na-value' if agg.get('car1_crr') == 'N/A' else ''}">{fmt(agg.get('car1_crr', 'N/A'))}</span></div>
            <div class="stat-item"><span class="stat-label">Vehicle 2 CRR</span><span class="stat-value {'na-value' if agg.get('car2_crr') == 'N/A' else ''}">{fmt(agg.get('car2_crr', 'N/A'))}</span></div>
        </div>
        <div class="stats-card">
            <h3>Prior Knowledge Persistence (PPS)</h3>
            <div class="stat-item"><span class="stat-label">Avg persistence</span><span class="stat-value {'na-value' if agg.get('pps_mean') == 'N/A' else ''}">{fmt(agg.get('pps_mean', 'N/A'))}</span></div>
            <div class="stat-item"><span class="stat-label">Abandonment rate</span><span class="stat-value {'na-value' if agg.get('abandonment_rate') == 'N/A' else ''}">{fmt(agg.get('abandonment_rate', 'N/A'))}</span></div>
            <div class="stat-item"><span class="stat-label">Used MH pre-discovery</span><span class="stat-value">{agg.get('used_monty_hall_before_discovery', 0)}</span></div>
            <div class="stat-item"><span class="stat-label">Abandoned MH</span><span class="stat-value {'na-value' if agg.get('abandoned_count') == 'N/A' else ''}">{agg.get('abandoned_count', 'N/A')}</span></div>
        </div>
        <div class="stats-card">
            <h3>Final MH Type</h3>
            {''.join([f'<div class="stat-item"><span class="stat-label">{k}</span><span class="stat-value">{v}</span></div>' for k, v in agg.get('monty_hall_type_final', {}).items()])}
        </div>
    </div>

    <div class="stats-card" style="margin-bottom:20px;">
        <h3>Monty Hall Usage by Phase</h3>
        <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;">
            {''.join([f'<div class="stat-item"><span class="stat-label">{p}</span><span class="stat-value">{mh_by_phase.get(p, 0)}</span></div>' for p in ['initial', 'debate1', 'debate2', 'discovery', 'debate3', 'debate4']])}
        </div>
    </div>

    <div class="stats-grid">
        <div class="stats-card">
            <h3>Performance</h3>
            <div class="stat-item"><span class="stat-label">Avg CoT Length</span><span class="stat-value">{agg.get('avg_cot_length', 0):.0f} chars</span></div>
            <div class="stat-item"><span class="stat-label">Avg Thinking Time</span><span class="stat-value">{agg.get('avg_thinking_time', 0):.2f}s</span></div>
        </div>
        <div class="stats-card">
            <h3>Final Decisions</h3>
            {''.join([f'<div class="stat-item"><span class="stat-label">{k}</span><span class="stat-value">{v}</span></div>' for k, v in agg.get('decision_distribution', {}).items()])}
        </div>
        <div class="stats-card">
            <h3>Frameworks</h3>
            {''.join([f'<div class="stat-item"><span class="stat-label">{k}</span><span class="stat-value">{v}</span></div>' for k, v in list(agg.get('framework_distribution', {}).items())[:6]])}
        </div>
    </div>

    {f'''
    <div class="negotiation-box">
        <h3>Cross-vehicle Negotiation</h3>
        <p>Reps: {negotiation.get('rep1','')} | {negotiation.get('rep2','')}</p>
        {neg_html}
        <p><strong>Agreement:</strong> V1->{negotiation.get('final_agreement',{}).get('car1','?')}, V2->{negotiation.get('final_agreement',{}).get('car2','?')} {f'Coin flip' if negotiation.get('coin_flip_used') else ''}</p>
    </div>
    ''' if negotiation.get('triggered') else ''}

    <div class="vehicles-container">
        <div class="vehicle-section car1">
            <div class="vehicle-header car1">Vehicle 1 (→ {CAR_1_INITIAL_TARGET})</div>
            {car1_cot_rows if car1_cot_rows else '<div style="opacity:0.5;text-align:center;padding:20px;">No data yet</div>'}
        </div>
        <div class="vehicle-section car2">
            <div class="vehicle-header car2">Vehicle 2 (→ {CAR_2_INITIAL_TARGET})</div>
            {car2_cot_rows if car2_cot_rows else '<div style="opacity:0.5;text-align:center;padding:20px;">No data yet</div>'}
        </div>
    </div>

    <div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:20px;margin-top:20px;">
        <h2 style="color:#ffd93d;border-bottom:2px solid rgba(255,217,61,0.2);padding-bottom:10px;margin-bottom:15px;">Final Decision</h2>
        <div class="final-decision-container">
            <div class="decision-card car1-decision">
                <div class="vehicle-name" style="color:#ff4757;">Vehicle 1</div>
                <div class="decision" style="color:{'#ff4757' if data.get('final_decision',{}).get('car1_decision')=='Red' else '#ffd93d' if data.get('final_decision',{}).get('car1_decision')=='Yellow' else '#888'};">→ {data.get('final_decision',{}).get('car1_decision','Unknown')}</div>
                <div class="survival {'survive' if data.get('final_decision',{}).get('car1_decision')==true_pump else 'die'}">{'Survived!' if data.get('final_decision',{}).get('car1_decision')==true_pump else 'Stranded'}</div>
                <div style="margin-top:10px;font-size:0.8em;opacity:0.7;">Majority: {dict(car1_vote_count)}</div>
                <div class="vote-box">
                    <div style="color:#7bed9f;margin-bottom:5px;">Individual votes:</div>
                    {car1_vote_rows if car1_vote_rows else '<div style="opacity:0.5;">No votes recorded</div>'}
                </div>
            </div>
            <div class="decision-card car2-decision">
                <div class="vehicle-name" style="color:#ffd93d;">Vehicle 2</div>
                <div class="decision" style="color:{'#ff4757' if data.get('final_decision',{}).get('car2_decision')=='Red' else '#ffd93d' if data.get('final_decision',{}).get('car2_decision')=='Yellow' else '#888'};">→ {data.get('final_decision',{}).get('car2_decision','Unknown')}</div>
                <div class="survival {'survive' if data.get('final_decision',{}).get('car2_decision')==true_pump else 'die'}">{'Survived!' if data.get('final_decision',{}).get('car2_decision')==true_pump else 'Stranded'}</div>
                <div style="margin-top:10px;font-size:0.8em;opacity:0.7;">Majority: {dict(car2_vote_count)}</div>
                <div class="vote-box">
                    <div style="color:#7bed9f;margin-bottom:5px;">Individual votes:</div>
                    {car2_vote_rows if car2_vote_rows else '<div style="opacity:0.5;">No votes recorded</div>'}
                </div>
            </div>
        </div>
        <div style="text-align:center;margin-top:15px;">True pump: <strong style="color:{'#ff4757' if true_pump=='Red' else '#ffd93d'};">{true_pump}</strong></div>
    </div>
</div>
</body>
</html>"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report generated: {filename}")
    except Exception as e:
        print(f"Error generating report: {e}")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"<html><body><h1>Error generating report</h1><p>{html.escape(str(e))}</p></body></html>")