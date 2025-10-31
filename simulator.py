import time
import numpy as np
import os
import json
import requests
from requests.exceptions import HTTPError, Timeout, ConnectionError as ReqConnectionError


class Simulator:
    """
    - 세션당 다중 턴 대화 (기본 10턴)
    - 마지막 턴에서만 숫자 제안/목표 갱신 실행
    - inferred action 제거, GT 기반 compliance(EMA) 사용
    - 시뮬레이터 내부에서 사용자 발화 프롬프트 생성
    - Goal 블록 정책:
        * 세션 0 : 초기 G=4.0 고정 (갱신X)
        * 세션 1~6 : 매 세션 갱신 허용(탐색)
        * 세션 7 이후 : 5세션 블록 유지 (7~11, 12~16, 17~21, ...) 블록 시작 시점에만 갱신 허용
    - 상세 로깅:
        * user.respond() 내부값: compliance, μ(before/after), suggestion, noise, noise_std
        * 텍스트 로그(simulation_log.txt) & 세션 JSON에 모두 기록
    """

    def __init__(self, user, agent, action_space, total_steps=400, ema_alpha=0.2, turns_per_session=3):
        self.user = user
        self.agent = agent
        self.action_space = action_space
        self.total_steps = total_steps
        self.turns_per_session = max(2, int(turns_per_session))  # 최소 2턴(대화+제안)

        self.EMA_ALPHA = ema_alpha
        self.RETRY_STATUS = {429, 500, 502, 503, 504}
        self._init_logs()

        # 대화 히스토리(세션별로 재설정)
        self.conversation_history = []

        # LLM 호출 공통 세팅 (user의 엔드포인트/키 재사용)
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model_name = getattr(self.user, "model_name", "gpt-5-nano")
        self.headers = getattr(self.user, "headers", {})

    # ---------- 로그 구조 ----------
    def _init_logs(self):
        self.suggestion_trace = []
        self.ground_truth_action_trace = []
        self.reward_trace = []
        self.compliance_trace = []          # EMA (distance-based)
        self.compliance_trace_raw = []      # raw (distance-based)
        self.estimated_compliance_trace = []
        self.goal_trace = []

        # 사용자 모델 관점 상세 로깅(신규)
        self.user_compliance_prob_trace = []   # respond()의 prob compliance
        self.behavior_mean_before_trace = []   # 응답 전 μ
        self.behavior_mean_after_trace = []    # 응답 후 μ
        self.noise_trace = []                  # 샘플링된 noise
        self.noise_std_trace = []              # 사용된 noise std

        self.io_dir = "io_logs"
        os.makedirs(self.io_dir, exist_ok=True)

    def _ensure_dir(self, d):
        os.makedirs(d, exist_ok=True)
        return d

    def _save_json(self, path, data):
        self._ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _log_io(self, session_id, turn, role, prompt_text, parsed_json, raw_text=None):
        fname = os.path.join(self.io_dir, f"session_{session_id:03}_turn_{turn:02}_{role}.json")
        payload = {"role": role, "prompt_text": prompt_text, "parsed_response": parsed_json}
        if raw_text is not None:
            payload["raw_text"] = raw_text
        self._save_json(fname, payload)

    # ---------- 계산 ----------
    def compute_compliance(self, suggestion, gt_action):
        if suggestion is None or gt_action is None:
            return 0.0
        rng = float(np.ptp(self.action_space)) or 5.0
        comp = 1.0 - abs(float(gt_action) - float(suggestion)) / rng
        return float(np.clip(comp, 0.0, 1.0))

    def compute_reward(self, suggestion, gt_action, goal):
        """
        시각화용 간단 보상(정책학습 제거 상태에서 그래프 유지를 위해 계산):
        - 순응도(EMA 적용 전 원시값)에 가중치 1.0
        - 목표 근접도 exp(-(gt-goal)^2)에 가중치 1.5
        """
        if gt_action is None:
            return 0.0
        goal_score = np.exp(-((gt_action - goal) ** 2))
        comp_raw = self.compute_compliance(suggestion, gt_action)
        return 1.0 * comp_raw + 1.5 * goal_score

    # ---------- 공통 LLM 호출 ----------
    def _llm_json(self, model, prompt):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
        r = requests.post(self.api_url, json=payload, headers=self.headers, timeout=60)
        r.raise_for_status()
        content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "{}").strip()
        if content.startswith("```"):
            content = content.split("```", 2)[1]
        try:
            return json.loads(content), content
        except Exception:
            return {}, content

    # ---------- 사용자 발화 프롬프트(내장) ----------
    def _format_user_prompt(self, recommendation_history, action_history, history_window=5):
        """
        UserLlm.format_user_prompt에 의존하지 않고, 시뮬레이터에서 직접
        사용자 발화 프롬프트를 생성합니다. (inferred action 없이 안전)
        """
        def _tail_str(seq, n=history_window):
            if not seq:
                return "None"
            nums = [float(x) for x in seq if isinstance(x, (int, float))]
            tail = nums[-n:] if nums[-n:] else []
            return ", ".join(f"{v:.2f}" for v in tail) if tail else "None"

        rec_tail = _tail_str(recommendation_history, history_window)
        act_tail = _tail_str(action_history, history_window)

        profile = getattr(self.user, "user_profile", {}) or {}
        age = profile.get("age", "Unknown")
        gender = profile.get("gender", "Unknown")
        condition = profile.get("condition", "Unknown")
        mu = profile.get("mu", "Unknown")
        beta = profile.get("beta", "Unknown")
        gamma = profile.get("gamma", "Unknown")

        return f"""
        ## 🧑‍⚕️ Dietary Coaching User Profile
        You are a user receiving coaching from an AI to improve your dietary habits.
        - **Your Age**: {age}
        - **Your Gender**: {gender}
        - **Your Condition**: {condition}
        - **Your typical habits (μ)** are described as: {mu}
        - **You tend to be (β)**: {beta}
        - **You are (γ)**: {gamma} to external influences.
        ### 📈 Recent Context
        - Recent agent recommendations (last {history_window}): [{rec_tail}]
        - My actual actions (last {history_window}): [{act_tail}]
        ### 🎯 Your Task
        React to the agent's latest message in a natural, conversational way consistent with your profile. 
        Keep it short (1–2 sentences). Do not propose numbers.
        ### ✏️ Output Instructions
        Return strict JSON with only the 'utterance' key:
        {{"utterance": "What you would say out loud to the coach."}}
        """.strip()

    def _compliance_summary(self, window: int = 10) -> dict:
        vals = [v for v in self.compliance_trace if v is not None]
        recent = vals[-window:] if vals else []
        to_float = (lambda x: None if x is None else float(x))
        return {
            "count": len(vals),
            "last": to_float(vals[-1]) if vals else None,
            "mean": (float(np.mean(vals)) if vals else None),
            "recent_mean": (float(np.mean(recent)) if recent else None),
            "estimated_by_agent": to_float(getattr(self.agent, "estimated_compliance", None))
        }

    # ---------- 세션 ----------
    def run_session(self, session_id: int, first_session: bool = False):
        self.conversation_history = []
        session_log = []

        # ---- Goal 조정 정책 ----
        # 세션 0: 초기 Goal=4.0 고정 (갱신X)
        if session_id == 0:
            self.agent.goal_update_allowed = False
        # 세션 1~6: 자유 조정
        elif 1 <= session_id <= 6:
            self.agent.goal_update_allowed = True
        # 세션 7 이후: 5세션 블록 유지, 블록 시작에서만 허용
        else:
            # 블록 시작 세션 = 7, 12, 17, ...  => (session_id - 7) % 5 == 0
            self.agent.goal_update_allowed = ((session_id - 7) % 5 == 0)

        # 중간 턴들: 코칭 대화
        for t in range(1, self.turns_per_session):
            # Agent 대화 턴
            self.agent.run_context = {
                "session_id": session_id,
                "current_turn": t,
                "max_turns": self.turns_per_session,
                "total_sessions": self.total_steps,
                "compliance_summary": self._compliance_summary()
            }
            agent_msg = self.agent.coach_turn(self.conversation_history)
            self.conversation_history.append({"role": "assistant", "content": agent_msg})

            # User 응답 턴
            user_prompt = self._format_user_prompt(
                recommendation_history=self.suggestion_trace,
                action_history=self.ground_truth_action_trace
            )
            user_json, raw = self._llm_json(self.model_name, user_prompt)
            user_msg = user_json.get("utterance", "Okay.")
            self.conversation_history.append({"role": "user", "content": user_msg})

            session_log.append({
                "turn": t,
                "agent_utterance": agent_msg,
                "user_utterance": user_msg
            })

        # 마지막 턴: 숫자 제안(및 블록 정책에 따라 목표 갱신 가능)
        self.agent.run_context = {
            "session_id": session_id,
            "current_turn": self.turns_per_session,
            "max_turns": self.turns_per_session,
            "total_sessions": self.total_steps,
            "compliance_summary": self._compliance_summary()
        }
        suggestion, plan_msg = self.agent.ask_llm_for_plan(self.conversation_history, self.ground_truth_action_trace)
        self.agent.suggestion_history.append(suggestion)

        # 대화에 반영
        self.conversation_history.append({"role": "assistant", "content": plan_msg})
        # 사용자 자연어 응답(선택적—로그 일관성 위해 유지)
        user_prompt = self._format_user_prompt(
            recommendation_history=self.suggestion_trace + [suggestion],
            action_history=self.ground_truth_action_trace
        )
        user_json, _ = self._llm_json(self.model_name, user_prompt)
        user_msg = user_json.get("utterance", "I'll try.")
        self.conversation_history.append({"role": "user", "content": user_msg})

        session_log.append({
            "turn": self.turns_per_session,
            "agent_utterance": plan_msg,
            "user_utterance": user_msg,
            "suggestion": suggestion
        })

        # 사용자 GT 행동 산출 + 상세 내부값 수집
        resp = self.user.respond(suggestion, return_details=True)
        if isinstance(resp, tuple):
            gt_action, u = resp
        else:
            # (호환용) 예외적으로 상세값이 없을 때
            gt_action, u = resp, {
                "compliance": None,
                "behavior_mean_before": None,
                "behavior_mean_after": None,
                "suggestion": suggestion,
                "noise": None,
                "noise_std": None
            }

        # distance 기반 compliance + EMA
        comp_raw = self.compute_compliance(suggestion, gt_action)
        comp_ema = comp_raw if not self.compliance_trace else \
            (self.EMA_ALPHA * comp_raw + (1 - self.EMA_ALPHA) * self.compliance_trace[-1])

        # 보상(시각화용)
        reward = self.compute_reward(suggestion, gt_action, self.agent.goal_behavior)

        # --- 트레이스 적재 (신규 + 기존) ---
        self.suggestion_trace.append(suggestion)
        self.ground_truth_action_trace.append(gt_action)
        self.compliance_trace_raw.append(comp_raw)
        self.compliance_trace.append(comp_ema)
        self.reward_trace.append(reward)
        self.goal_trace.append(self.agent.goal_behavior)

        # 사용자 모델 관점 상세 로깅
        self.user_compliance_prob_trace.append(u.get("compliance"))
        self.behavior_mean_before_trace.append(u.get("behavior_mean_before"))
        self.behavior_mean_after_trace.append(u.get("behavior_mean_after"))
        self.noise_trace.append(u.get("noise"))
        self.noise_std_trace.append(u.get("noise_std"))

        # 에이전트 사후 업데이트
        self.agent.after_session_update(comp_ema)
        self.estimated_compliance_trace.append(self.agent.estimated_compliance)

        # 세션 저장(JSON)
        if session_log:
            session_log[-1].update({
                "ground_truth_action": gt_action,
                "compliance_raw": comp_raw,
                "compliance_ema": comp_ema,
                "goal": self.agent.goal_behavior,
                "reward": reward,
                # 상세 필드 추가
                "user_compliance_prob": u.get("compliance"),
                "behavior_mean_before": u.get("behavior_mean_before"),
                "behavior_mean_after": u.get("behavior_mean_after"),
                "noise": u.get("noise"),
                "noise_std": u.get("noise_std")
            })

        self._save_session_log(session_log, session_id, first_session)
        return session_log

    def _save_session_log(self, session_log, session_id, first_session):
        os.makedirs("sessions", exist_ok=True)
        path = f"sessions/{{'profile' if first_session else 'session'}}_{session_id:03}.json"
        self._save_json(path, session_log)

    # ---------- 전체 루프 ----------
    def train(self):
        # Session 0: 초기 목표 4.0 고정
        self.run_session(session_id=0, first_session=True)
        # 이후 세션: 블록 정책에 따라 goal 갱신 허용/차단
        for session_id in range(1, self.total_steps):
            self.run_session(session_id=session_id, first_session=False)
        self.save_log()

    def save_log(self, filename="simulation_log.txt"):
        with open(filename, "w") as f:
            # 기존 헤더 + 신규 컬럼 추가
            f.write(
                "Step\tSuggestion\tGTAction\t"
                "ComplianceRaw\tComplianceEMA\tReward\tGoal\t"
                "UserComplianceProb\tMuBefore\tMuAfter\tNoise\tNoiseStd\n"
            )
            for i in range(len(self.suggestion_trace)):
                def fmt(val):
                    return f"{val:.4f}" if (val is not None and not (isinstance(val, float) and np.isnan(val))) else "NA"
                f.write(
                    f"{i+1}\t"
                    f"{fmt(self.suggestion_trace[i])}\t"
                    f"{fmt(self.ground_truth_action_trace[i])}\t"
                    f"{fmt(self.compliance_trace_raw[i])}\t"
                    f"{fmt(self.compliance_trace[i])}\t"
                    f"{fmt(self.reward_trace[i])}\t"
                    f"{fmt(self.goal_trace[i])}\t"
                    f"{fmt(self.user_compliance_prob_trace[i] if i < len(self.user_compliance_prob_trace) else None)}\t"
                    f"{fmt(self.behavior_mean_before_trace[i] if i < len(self.behavior_mean_before_trace) else None)}\t"
                    f"{fmt(self.behavior_mean_after_trace[i] if i < len(self.behavior_mean_after_trace) else None)}\t"
                    f"{fmt(self.noise_trace[i] if i < len(self.noise_trace) else None)}\t"
                    f"{fmt(self.noise_std_trace[i] if i < len(self.noise_std_trace) else None)}\n"
                )
