import os
import json
import math
from datetime import datetime
import numpy as np
import requests


class Agent:
    """
    경량 에이전트:
    - 정책/보상 수학 모델 제거, LLM(GPT-5 nano)이 대화/제안/목표조정을 판단
    - 초기 goal=4.0 고정, 이후 세션부터 유동적 갱신 허용
    - 순응도는 GT 기반 EMA로만 추정치 업데이트
    - 제안은 항상 최근 GT 행동보다 '조금 더 높게' 내되, EMA 기반 상한으로 급락(불순응) 방지
    """

    def __init__(
        self,
        action_space,
        user_age=None,
        user_gender=None,
        model_name="gpt-5-nano",
        ema_alpha=0.2,
        max_suggestion_step=0.5,  # 기본(하한) 연속 변화 제한
    ):
        self.action_space = np.array(action_space)
        self.user_profile = {"age": user_age, "gender": user_gender}

        # 목표: 초기 4.0 고정 (세션 0)
        self.goal_behavior = 4.0
        self.initial_goal_locked = True  # 세션 0 동안 잠금

        # 상태 추적
        self.estimated_compliance = 0.5
        self.ema_alpha = ema_alpha
        self.base_max_step = max_suggestion_step
        self.suggestion_history = []
        self.run_context = {}

        # LLM 설정
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    # ---------- 내부 보조 ----------
    def _update_estimated_compliance(self, compliance_ema):
        if compliance_ema is None:
            return
        lr = 0.6
        err = float(compliance_ema) - float(self.estimated_compliance)
        self.estimated_compliance = float(np.clip(self.estimated_compliance + lr * err, 0.0, 1.0))

    def _adaptive_max_step(self):
        """
        최근 EMA 순응도에 따라 연속 변화 제한 폭을 적응형으로 조정.
        EMA 0.6 → 약 0.3 / EMA 0.9 → 약 1.5
        """
        comp = float(self.run_context.get("compliance_summary", {}).get("recent_mean")
                     or self.estimated_compliance or 0.5)
        step = 0.3 + 2.2 * max(0.0, comp - 0.6)  # 선형 증가
        return float(np.clip(step, 0.3, 1.5))

    def _step_clamp(self, new_value, prev_value):
        if prev_value is None:
            return float(new_value)
        max_step = self._adaptive_max_step()
        lo, hi = prev_value - max_step, prev_value + max_step
        return float(np.clip(new_value, lo, hi))

    def _last_gt_action(self, recent_actions):
        """최근 GT action 중 마지막 유효값을 반환. 없으면 None."""
        for x in reversed(recent_actions):
            if isinstance(x, (int, float)) and not math.isnan(x):
                return float(x)
        return None

    def _growth_bounds(self, ema_recent: float | None):
        """
        '행동보다 위'로 제안할 때의 하한/상한 간격 설정.
        - min_gap: 항상 이만큼은 위로 (학습 추진)
        - max_gap: 이 이상 벌어지면 compliance 급락 위험 → 억제 (EMA 높을수록 넓힘)
        """
        ema = float(ema_recent) if ema_recent is not None else float(self.estimated_compliance or 0.5)
        min_gap = 0.10  # 최소 증가폭(항상 보장)
        # EMA 0.6 → 0.25, EMA 0.9 → 0.60 사이
        max_gap = 0.25 + (max(0.0, ema - 0.6) / 0.3) * (0.60 - 0.25)
        max_gap = float(np.clip(max_gap, 0.25, 0.60))
        return min_gap, max_gap

    # ---------- LLM 프롬프트 ----------
    def _format_llm_prompt_dialogue(self, history_str, planned_goal, compliance_summary):
        now = datetime.now()
        return f"""
You are a warm, practical dietary coach. Continue the conversation with ONE short message (1-2 sentences).
Do NOT give any numeric plan yet. Ask or reflect to help the user move forward.

Return strict JSON: {{"utterance": "..."}}.

Context:
- Location: Suwon-si, South Korea
- Day/Time: {now.strftime("%A")} {now.strftime("%I:%M %p")}
- Current Goal G: {planned_goal:.2f}
- Compliance (agent est, EMA): {compliance_summary.get('estimated_by_agent')}
- Conversation so far:
{history_str}
""".strip()

    def _format_llm_prompt_plan(self, history_str, planned_goal, compliance_summary, recent_suggestions, recent_actions):
        now = datetime.now()
        return f"""
You are a dietary behavior coach. Based on the whole conversation, propose ONE numeric suggestion S in [1.0, 5.0] and a brief supportive message.
You may also propose an updated goal G' in [1.0, 5.0] if it seems helpful now. If not, set it to null.

Return strict JSON: {{"suggestion": 3.2, "message": "...", "maybe_goal": 4.0}}

Context:
- Location: Suwon-si, South Korea
- Day/Time: {now.strftime("%A")} {now.strftime("%I:%M %p")}
- Current goal G: {planned_goal:.2f}
- Compliance (agent est, EMA): {compliance_summary.get('estimated_by_agent')}
- Recent suggestions: {recent_suggestions}
- Recent GT actions: {recent_actions}
- Conversation so far:
{history_str}

Guidance:
- Keep S realistic and gently progressive; avoid big jumps.
- If actions trend far below G for a while, consider lowering G slightly; if trending up, you may nudge it up.
""".strip()

    def _llm_json(self, prompt):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        r = requests.post(self.api_url, json=payload, headers=self.headers, timeout=60)
        r.raise_for_status()
        content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "{}").strip()
        if content.startswith("```"):
            content = content.split("```", 2)[1]
        try:
            return json.loads(content)
        except Exception:
            return {}

    # ---------- 외부에서 호출 ----------
    def coach_turn(self, conversation_history):
        """중간 대화 턴에서 호출 (숫자 제안 없이 코칭/질문만)"""
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in conversation_history]) or "No history."
        comp = self.run_context.get("compliance_summary", {}) or {}
        prompt = self._format_llm_prompt_dialogue(history_str, self.goal_behavior, comp)
        data = self._llm_json(prompt)
        utt = data.get("utterance", "How are you feeling about your routine today?")
        return utt

    def ask_llm_for_plan(self, conversation_history, recent_actions):
        """
        세션 마지막 턴에서 호출 (숫자 제안 S 및 (선택) 목표 갱신).
        - 제안은 항상 최근 GT action보다 '조금 위'에서 시작
        - EMA 기반 상한으로 compliance 급락 방지
        - 그 후 연속 변화 제한(적응형 스텝) 적용
        """
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in conversation_history]) or "No history."
        recent_suggestions = [round(float(x), 2) for x in self.suggestion_history[-10:]]
        recent_actions_rounded = [None if x is None else round(float(x), 2) for x in recent_actions[-10:]]
        comp = self.run_context.get("compliance_summary", {}) or {}

        prompt = self._format_llm_prompt_plan(
            history_str=history_str,
            planned_goal=self.goal_behavior,
            compliance_summary=comp,
            recent_suggestions=recent_suggestions,
            recent_actions=recent_actions_rounded
        )
        data = self._llm_json(prompt)

        # 1) LLM 제안 수신
        s = float(np.clip(float(data.get("suggestion", self.goal_behavior)), 1.0, 5.0))
        g = data.get("maybe_goal", None)
        g = None if g is None else float(np.clip(float(g), 1.0, 5.0))
        msg = data.get("message", "Let's take a small, doable step today.")

        # 2) '항상 행동보다 위' + 'EMA 상한' 적용
        last_gt = self._last_gt_action(recent_actions)
        if last_gt is not None:
            min_gap, max_gap = self._growth_bounds(comp.get("recent_mean"))
            lower = last_gt + min_gap
            upper = last_gt + max_gap
            s = float(np.clip(s, lower, upper))

        # 3) 연속 변화 제한(적응형 스텝)
        prev = self.suggestion_history[-1] if self.suggestion_history else None
        s = self._step_clamp(s, prev)

        # 4) 스텝 클램프 후에도 '행동보다 아래'면 최소 증가폭까지 재상향
        if last_gt is not None:
            min_gap, _ = self._growth_bounds(comp.get("recent_mean"))
            s = max(s, last_gt + min_gap)

        # 5) 목표 갱신 (초기 세션 이후만)
        if not self.initial_goal_locked and g is not None:
            self.goal_behavior = g

        # 6) 최종 범위 보정
        s = float(np.clip(s, 1.0, 5.0))
        return s, msg

    def after_session_update(self, compliance_ema):
        self._update_estimated_compliance(compliance_ema)

    def unlock_initial_goal(self):
        self.initial_goal_locked = False
