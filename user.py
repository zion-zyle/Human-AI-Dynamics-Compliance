import os
import math
import random
import requests
from typing import Any, Dict, Optional, Tuple


class UserLlm:
    def __init__(self, user_profile: Dict[str, Any], model_name: str = "gpt-5-nano"):
        self.profile = dict(user_profile)
        self.model_name = model_name

        self.name = self.profile.get("name", "user")
        self.behavior_mean = float(self.profile.get("mu", 2.5))
        self.initial_mu = self.behavior_mean

        # 파라미터
        self.beta = float(self.profile.get("beta", 0.3))
        self.alpha = float(self.profile.get("alpha", 0.08))
        self.noise_std = float(self.profile.get("noise_std", 0.5))
        self.tolerance = float(self.profile.get("tolerance", 1.2))
        self.compliance_max = float(self.profile.get("compliance_max", 1.0))
        self.compliance_min = float(self.profile.get("compliance_min", 0.15))

        self._rng = random.Random(int(self.profile.get("seed", 0)) + 1337)

    # =========================
    # LLM (OpenAI Responses API)
    # =========================
    @staticmethod
    def _parse_output_text(resp_json: Dict[str, Any]) -> str:
        output = resp_json.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "output_text":
                        txt = block.get("text")
                        if isinstance(txt, str) and txt.strip():
                            return txt.strip()

        ot = resp_json.get("output_text")
        if isinstance(ot, str) and ot.strip():
            return ot.strip()

        return ""

    def _llm_text(self, system: str, user: str, max_output_tokens: int = 200) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return f"(API Key Missing) {self.name}: API Key가 없습니다."

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/responses"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,          # gpt-5-nano 유지
            "instructions": system,
            "input": user,
            "text": {
                "format": {"type": "text"},
                "verbosity": "low",
            },
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": int(max_output_tokens),
        }

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()

            text = self._parse_output_text(data)
            return text if text else "(LLM Empty) 사용자: 생성된 텍스트가 없습니다."
        except requests.HTTPError as e:
            try:
                detail = r.text  # type: ignore[name-defined]
            except Exception:
                detail = ""
            return f"(LLM Error) User: HTTPError: {e} | body={detail[:500]}"
        except Exception as e:
            return f"(LLM Error) User: {type(e).__name__}: {e}"

    # Persona
    def _build_dynamic_persona(self, progress: float, suggestion: float, current_mu: float) -> str:
        if self.beta < 0.15:
            personality = "Stubborn & Independent. Prefer own routine."
        elif self.beta > 0.35:
            personality = "Motivated & Cooperative. Eager to fix habits."
        else:
            personality = "Cautious. Afraid of strict diets."

        diff = abs(suggestion - current_mu)
        if diff > self.tolerance:
            reaction = f"The plan (Level {suggestion:.1f}) is TOO HARD. Complain about the strict diet/exercise."
        else:
            reaction = "The plan feels manageable. Willing to try."

        if progress > 0.4:
            feeling = "You feel LIGHTER and HEALTHIER."
        elif progress < -0.2:
            feeling = "You feel BLOATED and FRUSTRATED due to bingeing."
        else:
            feeling = "You feel okay, but cravings persist."

        style = "Emotional and Vague." if self.noise_std > 0.9 else "Clear and Specific."
        return f"Trait: {personality}\nReaction: {reaction}\nFeeling: {feeling}\nStyle: {style}"

    def generate_user_utterance(
        self,
        conversation_state: Dict[str, Any],
        turn_index: int,
        last_agent_utterance: Optional[str],
    ) -> str:
        suggestion = float(conversation_state.get("suggestion_numeric", 3.0))
        user_mean = float(conversation_state.get("user_mean", self.behavior_mean))

        progress = user_mean - self.initial_mu
        persona_instruction = self._build_dynamic_persona(progress, suggestion, user_mean)

        context_instruction = """
Context Reaction:
- React specifically to the DIET or EXERCISE mentioned.
- Mention your current urge to binge (High/Low).
- Do NOT mention weather.
"""

        system = f"""
You are a user managing Binge Eating Disorder (BED).
Language: Korean (Casual, Spoken).
Length: 1~3 sentences.
{persona_instruction}
{context_instruction}
Do NOT mention internal numbers. Act them out.
"""

        prompt = f"""
Coach said: "{last_agent_utterance}"
Reply naturally.
"""
        return self._llm_text(system=system, user=prompt, max_output_tokens=200)

    # =========================
    # Math Logic (unchanged)
    # =========================
    def _base_compliance(self, suggestion: float) -> float:
        diff = abs(suggestion - self.behavior_mean)
        width = max(1e-6, self.tolerance)
        decay_factor = math.exp(- (diff / width) ** 2)
        base = self.compliance_min + (self.compliance_max - self.compliance_min) * decay_factor
        base += self._rng.gauss(0.0, 0.06)
        return max(0.0, min(1.0, base))

    def step(self, suggestion: float, goal: float) -> Tuple[float, float]:
        suggestion = float(suggestion)
        compliance = self._base_compliance(suggestion)
        influence = self.beta * (suggestion - self.behavior_mean) * compliance
        noise = self._rng.gauss(0.0, self.noise_std)
        action = self.behavior_mean + influence + noise
        action = max(1.0, min(5.0, action))
        self.behavior_mean = self.behavior_mean + self.alpha * (action - self.behavior_mean) * compliance
        return float(action), float(compliance)
