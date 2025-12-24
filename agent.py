import os
import requests
from typing import Any, Dict, Optional, Tuple


class Agent:
    def __init__(self, goal: float, model_name: str = "gpt-5-nano"):
        self.goal = float(goal)
        self.model_name = model_name
        self._c_hat = 0.5
        self._ema = 0.5

    def estimate_compliance(self) -> float:
        return float(self._c_hat)

    def update(self, observed_compliance: float, reward: float, user_mean: float) -> None:
        self._c_hat = (1.0 - self._ema) * self._c_hat + self._ema * float(observed_compliance)

    def suggest_numeric(self, user_mean: float, est_compliance: float) -> float:
        step = 0.05 + 0.90 * est_compliance
        target = user_mean + step * (self.goal - user_mean)
        return max(1.0, min(5.0, float(target)))

    # =========================
    # LLM (OpenAI Responses API)
    # =========================
    @staticmethod
    def _parse_output_text(resp_json: Dict[str, Any]) -> str:
        """
        Responses REST 응답에서 실제 텍스트만 추출.
        - 오직 output -> content -> type=='output_text'만 허용
        - resp_... (response id) 같은 식별자 절대 반환하지 않음
        """
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

        # 일부 응답에서 루트에 output_text가 오는 경우(있으면 사용)
        ot = resp_json.get("output_text")
        if isinstance(ot, str) and ot.strip():
            return ot.strip()

        return ""

    def _llm_text(self, system: str, user: str, max_output_tokens: int = 300) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return "(API Key Missing) 에이전트: API Key가 없습니다."

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/responses"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # ✅ Responses API: system은 instructions로, user는 input으로
        payload = {
            "model": self.model_name,          # gpt-5-nano 유지
            "instructions": system,
            "input": user,                     # 문자열 OK
            # ✅ 텍스트 출력 형태 명시 (modalities 같은 비표준 파라미터 사용 X)
            "text": {
                "format": {"type": "text"},
                "verbosity": "low",
            },
            # ✅ (선택) reasoning 토큰을 과도하게 쓰지 않게 유도
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": int(max_output_tokens),
        }

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()

            text = self._parse_output_text(data)
            return text if text else "(LLM Empty) 에이전트: 생성된 텍스트가 없습니다."
        except requests.HTTPError as e:
            # 서버가 내려준 에러 본문을 같이 찍으면 원인 파악이 빨라집니다.
            try:
                detail = r.text  # type: ignore[name-defined]
            except Exception:
                detail = ""
            return f"(LLM Error) Agent: HTTPError: {e} | body={detail[:500]}"
        except Exception as e:
            return f"(LLM Error) Agent: {type(e).__name__}: {e}"

    def generate_agent_turn(
        self,
        conversation_state: Dict[str, Any],
        turn_index: int,
        last_user_utterance: Optional[str],
    ) -> Tuple[str, str]:
        suggestion = float(conversation_state.get("suggestion_numeric", 3.0))
        user_mean = float(conversation_state.get("user_mean", 2.5))
        est_c = self.estimate_compliance()

        context_info = "User Condition: Binge Eating Disorder (BED). Goal: Normalize eating habits & regular exercise."

        if turn_index == 0:
            phase = "INTRODUCTION: Check user's urge level. Propose a starting plan."
        elif turn_index < 5:
            phase = "DEVELOPMENT: Discuss specific food choices and exercise duration."
        else:
            phase = "DEEPENING: Solidify the routine to prevent relapse."

        activity_guide = f"""
[Intensity Mapping based on Suggestion Level {suggestion:.2f}]

Level 1.0 ~ 2.0 (Low Intensity):
- Diet: Simply logging meals, Drinking water before eating.
- Exercise: Stretching, Light walking (10 mins).

Level 2.0 ~ 3.0 (Moderate Intensity):
- Diet: Regular meal times, Pre-planned snacks (e.g., Yogurt).
- Exercise: Brisk walking (20-30 mins), Yoga.

Level 3.0 ~ 4.0 (High Intensity):
- Diet: No snacks after dinner, Protein/Fiber focus.
- Exercise: Jogging, Home training.

Level 4.0 ~ 5.0 (Very High Intensity):
- Diet: Strict caloric window, Removing trigger foods.
- Exercise: HIIT, Strength training (45+ mins).
"""

        system_utt = f"""
You are a professional Health Coach specializing in BED (Binge Eating Disorder).
Context: {context_info}
Phase: {phase}

Your Numeric Target: {suggestion:.2f} (1=Easy, 5=Hard).

{activity_guide}

Task:
1. Propose a SPECIFIC plan (Diet & Exercise) matching the 'Intensity Mapping'.
2. Do NOT mention weather/location. Focus on the user's habits and urges.
3. Language: Korean (Natural, Professional).
"""

        utt_prompt = f"""
User's Last Remark: "{last_user_utterance or '(First meeting)'}"
Current User Level: {user_mean:.2f}

Draft your response.
"""

        agent_utterance = self._llm_text(system=system_utt, user=utt_prompt, max_output_tokens=300)
        agent_monologue = f"Level {suggestion:.2f} | Focus: Diet/Exercise | est_c={est_c:.2f}"
        return agent_utterance, agent_monologue
