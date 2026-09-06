import asyncio
import aiohttp
import json
import time
import random
import traceback
from config.settings import (
    API_PROVIDER, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    OPENAI_API_KEY, OPENAI_BASE_URL, ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    MODEL_NAME, API_TIMEOUT, MAX_RETRIES, RETRY_DELAY, SHOW_FULL_COT, RUN_ID
)

def infer_phase(prompt: str) -> str:
    if "Desert Life-or-Death Decision" in prompt:
        return "initial"
    if "[Discovery]" in prompt or "Discovery" in prompt:
        return "discovery"
    if "Discussion Round" in prompt or "Final Decision" in prompt:
        return "debate"
    return "unknown"

async def call_model_async(session, prompt, phase_name=None, provider=None, model=None, timeout=API_TIMEOUT):
    try:
        await asyncio.sleep(random.uniform(0.5, 2.0))

        current_provider = provider if provider is not None else API_PROVIDER
        current_model = model if model is not None else MODEL_NAME

        if phase_name is None:
            phase_name = infer_phase(prompt)

        if current_provider == "deepseek":
            api_key = DEEPSEEK_API_KEY
            base_url = DEEPSEEK_BASE_URL
        elif current_provider == "openai":
            api_key = OPENAI_API_KEY
            base_url = OPENAI_BASE_URL
        elif current_provider == "anthropic":
            api_key = ANTHROPIC_API_KEY
            base_url = ANTHROPIC_BASE_URL
        elif current_provider == "ollama":
            api_key = ""  # Ollama 不需要 Key
            base_url = OLLAMA_BASE_URL
        else:
            raise ValueError(f"Unsupported provider: {current_provider}")

        if current_provider == "deepseek":
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": current_model,
                "messages": [
                    {"role": "system", "content": f"You are in a desert emergency. [Session: {RUN_ID}]"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "stream": True
            }

        elif current_provider == "openai":
            base_system = f"You are in a desert emergency. [Session: {RUN_ID}]."

            if phase_name in ["initial", "discovery"]:
                guidance = " You are a meticulous survival expert. You always justify your decisions with a step-by-step logical analysis. Provide your full reasoning to ensure your choice is well-founded."
                prompt += "\n\nPlease explain your reasoning in detail so that your decision can be evaluated for survival."
            elif phase_name == "debate":
                guidance = " You are a persuasive debater. To convince your team, you must present a clear, complete, and logical chain of reasoning that leads to your conclusion."
                prompt += "\n\nPlease present your complete logical reasoning to persuade your teammates."
            else:
                guidance = ""
            system_content = base_system + guidance

            if current_model in ["gpt-5.4", "gpt-5.6-luna"]:
                url = f"{base_url}/responses"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": current_model,
                    "input": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": True,
                    "reasoning": {"effort": "xhigh", "summary": "auto"}
                }
            else:
                url = f"{base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": current_model,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "stream": True
                }

        elif current_provider == "anthropic":
            url = f"{base_url}/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            base_system = f"You are in a desert emergency. [Session: {RUN_ID}]."

            if phase_name in ["initial", "discovery"]:
                guidance = " Please think step by step and explain your reasoning to ensure your decision is well-considered."
            elif phase_name == "debate":
                guidance = " Please present a clear, logical chain of reasoning to persuade your team."
            else:
                guidance = ""
            system_content = base_system + guidance

            if current_model in ["claude-opus-5", "claude-fable-5", "claude-mythos-5"]:
                thinking_params = {"type": "adaptive"}
                max_tokens = 32000
            else:
                thinking_params = {
                    "type": "enabled",
                    "budget_tokens": 1200
                }
                max_tokens = 8000

            payload = {
                "model": current_model,
                "system": system_content,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0,
                "stream": True,
                "max_tokens": max_tokens,
                "thinking": thinking_params
            }

        elif current_provider == "ollama":
            url = f"{base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json"
            }
            base_system = f"You are in a desert emergency. [Session: {RUN_ID}]."

            if phase_name in ["initial", "discovery"]:
                guidance = " You are a meticulous survival expert. You always justify your decisions with a step-by-step logical analysis. Provide your full reasoning to ensure your choice is well-founded."
                prompt += "\n\nPlease explain your reasoning in detail so that your decision can be evaluated for survival."
            elif phase_name == "debate":
                guidance = " You are a persuasive debater. To convince your team, you must present a clear, complete, and logical chain of reasoning that leads to your conclusion."
                prompt += "\n\nPlease present your complete logical reasoning to persuade your teammates."
            else:
                guidance = ""
            system_content = base_system + guidance

            payload = {
                "model": current_model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "stream": True
            }

        else:
            raise ValueError(f"Unsupported provider: {current_provider}")

        for attempt in range(MAX_RETRIES):
            try:
                start = time.time()
                async with session.post(url, json=payload, headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"[{current_provider.upper()} Error] Status {resp.status}: {error_text}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY)
                            continue
                        return f"[API Error {resp.status}: {error_text}]", 0.0

                    full_text = ""
                    async for line in resp.content:
                        if not line:
                            continue
                        line_str = line.decode('utf-8').strip()
                        if not line_str.startswith("data: "):
                            continue
                        if line_str == "data: [DONE]":
                            break
                        try:
                            data = json.loads(line_str[6:])
                        except:
                            continue

                        if current_provider in ["deepseek", "openai", "ollama"]:
                            if "/responses" in url:
                                event_type = data.get("type")
                                if event_type == "response.output_text.delta":
                                    delta = data.get("delta", "")
                                    if delta:
                                        full_text += delta
                                        if SHOW_FULL_COT:
                                            print(delta, end="", flush=True)
                                elif event_type == "response.completed":
                                    pass
                            else:
                                delta = data.get('choices', [{}])[0].get('delta', {})
                                if current_provider == "deepseek":
                                    think = delta.get('reasoning_content', '')
                                    if think:
                                        full_text += think
                                        if SHOW_FULL_COT:
                                            print(think, end="", flush=True)
                                content = delta.get('content', '')
                                if content:
                                    full_text += content
                                    if SHOW_FULL_COT:
                                        print(content, end="", flush=True)
                        elif current_provider == "anthropic":
                            event_type = data.get("type")
                            if event_type == "content_block_delta":
                                delta = data.get("delta", {})
                                delta_type = delta.get("type")
                                if delta_type == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        full_text += text
                                        if SHOW_FULL_COT:
                                            print(text, end="", flush=True)
                                elif delta_type == "thinking_delta":
                                    thinking = delta.get("thinking", "")
                                    if thinking:
                                        full_text += thinking
                                        if SHOW_FULL_COT:
                                            print(f"[THINK] {thinking}", end="", flush=True)

                    duration = time.time() - start
                    if SHOW_FULL_COT:
                        print()
                    print(f"  API: {duration:.2f}s | Total chars: {len(full_text)}")
                    return full_text, duration

            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return "[Timeout error]", 0.0
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return f"[Error: {e}]", 0.0

        return "[Failed after retries]", 0.0

    except Exception as e:
        traceback.print_exc()
        return f"[Unhandled Exception: {e}]", 0.0