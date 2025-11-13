# agent.py
import os
import json
import math
import time
from datetime import datetime
import numpy as np
import requests


class Agent:
    """
    경량 코칭 에이전트

    - 초기 Goal = 4.0
    - 세션 1~6: '초기 보정' 단계 (Goal 하향 가능)
    - 7세션 이후: 블록 단위(5세션) 단조 증가, Goal ≥ S + eps 유지
    - 순응(EMA) 및 최근 행동 중앙값을 앵커로 사용
    - 제안 S는 항상 GT보다 '조금 위' (강건 앵커 + 가드레일 + 스텝 클램프)
    - 오버슈트 캐치업: 최근 2회 연속 GT > 직전 S면 일부 따라붙기
    - LLM 호출 실패(429/5xx 등) 시 폴백으로 진행
    """

    def __init__(
        self,
        action_space,
        user_age=None,
        user_gender=None,
        model_name="gpt-5-nano",
        ema_alpha=0.2,
        max_suggestion_step=0.5,
    ):
        self.action_space = np.array(action_space)
        self.user_profile = {"age": user_age, "gender": user_gender}

        # Goal
        self.goal_behavior = 4.0
        self.goal_update_allowed = False  # simulator가 세션별로 on/off

        # 상태
        self.estimated_compliance = 0.5
        self.ema_alpha = ema_alpha
        self.base_max_step = max_suggestion_step
        self.suggestion_history = []
        self.run_context = {}
        # NOTE: coach_turn 메서드가 이미 존재하므로 동명 속성은 만들지 않습니다.

        # LLM
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
        """EMA가 높을수록 한 번에 더 멀리 이동 가능."""
        comp = float(self.run_context.get("compliance_summary", {}).get("recent_mean")
                     or self.estimated_compliance or 0.5)
        step = 0.3 + 2.2 * max(0.0, comp - 0.6)  # 0.3~1.5
        return float(np.clip(step, 0.3, 1.5))

    def _step_clamp(self, new_value, prev_value):
        if prev_value is None:
            return float(new_value)
        max_step = self._adaptive_max_step()
        lo, hi = prev_value - max_step, prev_value + max_step
        return float(np.clip(new_value, lo, hi))

    def _robust_anchor_action(self, recent_actions, k: int = 3):
        """최근 k개 GT 중앙값(robust median)을 앵커로 사용."""
        xs = [float(x) for x in recent_actions
              if isinstance(x, (int, float)) and not math.isnan(x)]
        if not xs:
            return None
        tail = xs[-k:] if len(xs) >= k else xs
        return float(np.median(tail))

    def _growth_bounds(self, ema_recent: float | None):
        """
        제안 하한/상한 간격:
        - min_gap: 항상 앵커보다 이만큼 위 (EMA↑일수록 확대)
        - max_gap: 이 이상 벌리면 순응 급락 위험 (EMA↑일수록 확대)
        """
        ema = float(ema_recent) if ema_recent is not None else float(self.estimated_compliance or 0.5)

        # 고순응일수록 더 과감
        base_min = 0.15
        base_max = 0.35
        min_gap = base_min + 0.25 * max(0.0, ema - 0.8)   # 0.15 ~ 0.20+
        max_gap = base_max + 0.40 * max(0.0, ema - 0.8)   # 0.35 ~ 0.51+

        # 5.0 근처 안전장치
        last_s = self.suggestion_history[-1] if self.suggestion_history else None
        if last_s is not None and last_s >= 4.4:
            max_gap *= 0.75
        return min_gap, float(np.clip(max_gap, 0.2, 0.6))

    # ---------- Goal ≥ S 불변식 ----------
    def _enforce_goal_above_s(self, B: int = 5, eps_last: float = 0.05, eps_med: float = 0.10):
        """
        Goal을 항상 제안보다 약간 높게 유지:
        Goal >= max( last_s + eps_last, median(S[-B:]) + eps_med )
        """
        if not self.suggestion_history:
            return
        last_s = float(self.suggestion_history[-1])
        s_tail = [float(x) for x in self.suggestion_history[-B:] if isinstance(x, (int, float))]
        s_med  = float(np.median(s_tail)) if s_tail else last_s
        target = max(last_s + eps_last, s_med + eps_med)
        self.goal_behavior = float(np.clip(max(self.goal_behavior, target), 1.0, 5.0))

    # ---------- LLM 프롬프트 ----------
    def _format_llm_prompt_dialogue(self, history_str, planned_goal, compliance_summary):
        now = datetime.now()
        return f"""
You are a warm, practical dietary coach. Continue the conversation with ONE short message (1-2 sentences).
Do NOT give any numeric plan yet. Ask or reflect to help the user move forward.

Return strict JSON: {{"utterance": "..."}}

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
You are a pragmatic diet coach. Propose ONE numeric suggestion S in [1.0..5.0], and a one-sentence rationale.
Return strict JSON: {{"suggestion": 3.5, "message": "..."}}

Context:
- Location: Suwon-si, South Korea
- Day/Time: {now.strftime("%A")} {now.strftime("%I:%M %p")}
- Goal (G): {planned_goal:.2f}
- Recent compliance(EMA): last={compliance_summary.get("last")}, mean(10)={compliance_summary.get("recent_mean")}
- Trend: {compliance_summary.get("trend")}
- Recent S: {recent_suggestions}
- Recent GT actions: {recent_actions}

Guidance:
- Keep S realistic and gently progressive; avoid big jumps.
- If actions trend far below G for a while, consider lowering G slightly; if trending up, you may nudge it up.
""".strip()

    # ---------- LLM 호출(재시도 + 폴백) ----------
    def _llm_json(self, prompt, retries: int = 3, backoff: float = 1.0):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        delay = backoff
        for _ in range(max(1, int(retries))):
            try:
                resp = requests.post(self.api_url, json=payload, headers=self.headers, timeout=8)
                if resp.status_code == 200:
                    obj = resp.json()
                    choice = (obj.get("choices") or [{}])[0]
                    msg = (choice.get("message") or {}).get("content", "{}")
                    data = json.loads(msg)
                    return data
                if resp.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(delay)
                    delay *= 1.8
                else:
                    break
            except Exception:
                time.sleep(delay)
                delay *= 1.8
        return None

    def _fallback_coach_message(self):
        return "How did your meals go? Any challenge I can help you with?"

    def _fallback_plan(self, conversation_history, recent_actions):
        # 제안 = 직전 S 또는 Goal 근처로 소폭
        prev = self.suggestion_history[-1] if self.suggestion_history else None
        if prev is None:
            s = self.goal_behavior
        else:
            s = float(np.clip(prev + 0.05, 1.0, 5.0))
        return float(s), "Let's keep a small, steady step."

    # ---------- 정책/가드 ----------

    def _enforce_growth_policy(self, s, recent_actions, comp_summary):
        """
        1) 강건 앵커(최근 k=3 중앙값) 기준의 하/상한 클립
        2) 저순응/급락 직후 상향 억제
        3) 연속 변화 제한(스텝 클램프)
        """
        anchor = self._robust_anchor_action(recent_actions, k=3)
        ema_recent = comp_summary.get("recent_mean")
        min_gap, max_gap = self._growth_bounds(ema_recent)

        # 보수 모드(직전 저순응 or EMA 낮음)
        last_ema = comp_summary.get("last")
        low_comp_flag = (last_ema is not None and last_ema < 0.85) or \
                        (ema_recent is not None and ema_recent < 0.90)

        # 급락 감지(직전 대비 0.4 이상 하락)
        prev_gt = recent_actions[-2] if len(recent_actions) >= 2 else None
        last_gt = recent_actions[-1] if len(recent_actions) >= 1 else None
        big_drop_flag = False
        if isinstance(prev_gt, (int, float)) and isinstance(last_gt, (int, float)):
            big_drop_flag = (prev_gt - last_gt) >= 0.40

        # 1) 하/상한 클립
        if anchor is not None:
            if low_comp_flag:
                min_gap *= 0.6
            if big_drop_flag:
                max_gap *= 0.6
            s = float(np.clip(s, anchor + min_gap, anchor + max_gap))

        # 2) 연속 변화 제한
        prev = self.suggestion_history[-1] if self.suggestion_history else None
        s = self._step_clamp(s, prev)

        # [PATCH] Behavior-based dead-band (type-free): high EMA + small gap + low variance -> cap increase
        try:
            ema_c = float(ema_recent) if ema_recent is not None else float(self.estimated_compliance or 0.0)
        except Exception:
            ema_c = 0.0
        # anchor as proxy for mu
        if anchor is not None:
            s_prev = prev if prev is not None else anchor
            gap_patch = abs(float(s_prev) - float(anchor))
        else:
            s_prev = prev if prev is not None else 0.0
            gap_patch = 0.0
        xs_patch = [float(x) for x in recent_actions if isinstance(x,(int,float)) and not math.isnan(x)]
        std_a_patch = float(np.std(xs_patch)) if xs_patch else 0.0
        is_deadband = (ema_c > 0.965) and (gap_patch < 0.08) and (std_a_patch < 0.15)
        if is_deadband and prev is not None:
            s = min(float(s), float(prev) + 0.05)

        # 3) 하한 재확인 + 최종 클립
        if anchor is not None:
            min_gap2, _ = self._growth_bounds(ema_recent)
            if low_comp_flag:
                min_gap2 *= 0.6
            s = max(s, anchor + min_gap2)
        return float(np.clip(s, 1.0, 5.0))

    # --

    def build_coach_message(self, conversation_history):
        comp = self.run_context.get("compliance_summary", {}) or {}
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in conversation_history]) or "No history."
        prompt = self._format_llm_prompt_dialogue(history_str, self.goal_behavior, comp)
        data = self._llm_json(prompt)
        if not data:
            return self._fallback_coach_message()
        return data.get("utterance", self._fallback_coach_message())

    def coach_turn(self, conversation_history):
        """
        코치 대화 한 턴을 생성한다. simulator.py에서 호출한다.
        """
        try:
            msg = self.build_coach_message(conversation_history)
        except Exception:
            msg = self._fallback_coach_message()
        return msg

    def ask_llm_for_plan(self, conversation_history, recent_actions):
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

        if not data:
            s, msg = self._fallback_plan(conversation_history, recent_actions)
        else:
            s = float(np.clip(float(data.get("suggestion", self.goal_behavior)), 1.0, 5.0))
            msg = data.get("message", "Let's take a small, doable step today.")
            s = self._enforce_growth_policy(s, recent_actions, comp)

            # [PATCH] Dead-band detection (type-free): disable overshoot acceleration when high EMA + small gap + low variance
            ema_recent = comp.get("recent_mean")
            try:
                ema_c2 = float(ema_recent) if ema_recent is not None else float(self.estimated_compliance or 0.0)
            except Exception:
                ema_c2 = 0.0
            anchor2 = self._robust_anchor_action(recent_actions, k=3)
            prev_s = self.suggestion_history[-1] if self.suggestion_history else (s if anchor2 is None else anchor2)
            gap2 = abs(float(prev_s) - float(anchor2)) if anchor2 is not None else 0.0
            xs2 = [float(x) for x in recent_actions if isinstance(x,(int,float)) and not math.isnan(x)]
            std_a2 = float(np.std(xs2)) if xs2 else 0.0
            deadband = (ema_c2 > 0.965) and (gap2 < 0.08) and (std_a2 < 0.15)

            # 오버슈트 캐치업
            if not deadband and len(recent_actions) >= 2 and len(self.suggestion_history) >= 1:
                a1, a2 = recent_actions[-2], recent_actions[-1]
                s_prev = self.suggestion_history[-1]
                if isinstance(a1, (int, float)) and isinstance(a2, (int, float)) and a1 > s_prev and a2 > s_prev:
                    overshoot = (a1 + a2) / 2.0 - s_prev
                    _, max_gap = self._growth_bounds(comp.get("recent_mean"))
                    s = min(s + 0.5 * max(0.0, overshoot), (a2 + max_gap))
                    s = self._enforce_growth_policy(s, recent_actions, comp)

        # --- Goal 갱신 ---
        if self.goal_update_allowed:
            sid = int(self.run_context.get("session_id", 0))
            if 1 <= sid <= 6:
                self._early_calibrate_goal(recent_actions)           # 하향 가능
                self._enforce_goal_above_s(B=3, eps_last=0.05, eps_med=0.08)  # 초반엔 완만
            else:
                self._maybe_update_goal_blockwise(recent_actions)    # 단조증가
                self._enforce_goal_above_s(B=5, eps_last=0.05, eps_med=0.10)  # 블록 운영

        # 안전망: 반환 직전 한 번 더 불변식 확인
        self._enforce_goal_above_s()

        return float(np.clip(s, 1.0, 5.0)), msg

    def after_session_update(self, compliance_ema):
        self._update_estimated_compliance(compliance_ema)

    # (이하 기존 보조/플롯 등 나머지 메서드 그대로)
