import os
import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agent import Agent
from user import UserLlm
from visualization import plot_simulation


@dataclass
class StepResult:
    suggestion: float
    action: float
    reward: float
    compliance: float


class Simulator:
    def __init__(
        self,
        user_profile: Dict[str, Any],
        goal: float,
        n_sessions: int,
        turns_per_session: int,
        seed: int,
        out_dir: str,
        model_name: str = "gpt-5-nano",
    ):
        self.user_profile = dict(user_profile)
        self.goal = float(goal)
        self.n_sessions = int(n_sessions)
        self.turns_per_session = int(turns_per_session)
        self.seed = int(seed)
        self.out_dir = Path(out_dir)
        self.model_name = model_name

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = self.out_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        random.seed(self.seed)

        # 핵심 객체
        self.user = UserLlm(self.user_profile, model_name=self.model_name)
        self.agent = Agent(goal=self.goal, model_name=self.model_name)

        # 시계열 기록(플롯용)
        self.suggestions: List[float] = []
        self.actions: List[float] = []
        self.rewards: List[float] = []
        self.compliances: List[float] = []
        self.estimated_compliances: List[float] = []  # agent가 추정한 compliance(진단용)

    def _reward(self, action: float) -> float:
        # 목표에 가까울수록 reward 높음 (0~1 근사)
        err = abs(self.goal - action)
        return max(0.0, 1.0 - (err / 4.0))

    def run_one_session(self, session_id: int) -> StepResult:
        # 1) agent가 numeric suggestion 결정 (1~5)
        est_c = self.agent.estimate_compliance()
        suggestion = self.agent.suggest_numeric(user_mean=self.user.behavior_mean, est_compliance=est_c)

        # 2) session 내 turn 루프: 대화 생성(LLM)
        turn_logs: List[Dict[str, Any]] = []
        conversation_state = {
            "session_id": session_id,
            "goal": self.goal,
            "user_name": self.user_profile.get("name", "user"),
            "user_mean": self.user.behavior_mean,
            "suggestion_numeric": suggestion,
        }

        last_user_utterance = None
        for t in range(self.turns_per_session):
            agent_utt, agent_mono = self.agent.generate_agent_turn(
                conversation_state=conversation_state,
                turn_index=t,
                last_user_utterance=last_user_utterance,
            )
            user_utt = self.user.generate_user_utterance(
                conversation_state=conversation_state,
                turn_index=t,
                last_agent_utterance=agent_utt,
            )

            turn_logs.append(
                {
                    "turn": t,
                    "agent_utterance": agent_utt,
                    "agent_monologue": agent_mono,
                    "user_utterance": user_utt,
                    "suggestion_numeric": suggestion,  # ✅ 누락 방지
                }
            )
            last_user_utterance = user_utt

        # 3) 대화가 끝난 뒤, user action/ compliance 업데이트(수치 시뮬)
        action, compliance = self.user.step(suggestion=float(suggestion), goal=self.goal)

        # 4) reward 계산 및 agent 업데이트
        reward = self._reward(action)
        self.agent.update(observed_compliance=compliance, reward=reward, user_mean=self.user.behavior_mean)

        # 5) session JSON 저장 (✅ session_023처럼 list of turns)
        session_path = self.sessions_dir / f"session_{session_id:03d}.json"
        session_path.write_text(json.dumps(turn_logs, ensure_ascii=False, indent=2), encoding="utf-8")

        # 6) summary도 같이 저장(원하면)
        summary = {
            "session_id": session_id,
            "profile_name": self.user_profile.get("name", "user"),
            "goal": self.goal,
            "turns_per_session": self.turns_per_session,
            "suggestion": float(suggestion),
            "user_action": float(action),
            "reward": float(reward),
            "compliance": float(compliance),
            "estimated_compliance": float(est_c),
            "user_behavior_mean_after": float(self.user.behavior_mean),
            "timestamp": time.time(),
        }
        (self.sessions_dir / f"session_{session_id:03d}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return StepResult(suggestion=suggestion, action=action, reward=reward, compliance=compliance)

    def train(self) -> None:
        for s in range(self.n_sessions):
            res = self.run_one_session(session_id=s)

            self.suggestions.append(res.suggestion)
            self.actions.append(res.action)
            self.rewards.append(res.reward)
            self.compliances.append(res.compliance)
            self.estimated_compliances.append(self.agent.estimate_compliance())

        # 전체 요약 저장
        final = {
            "profile_name": self.user_profile.get("name", "user"),
            "n_sessions": self.n_sessions,
            "turns_per_session": self.turns_per_session,
            "goal": self.goal,
            "final_user_behavior_mean": float(self.user.behavior_mean),
        }
        (self.out_dir / "final_summary.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 플롯
        plot_path = self.out_dir / f"{self.user_profile.get('name','user')}_simulation_plot.png"
        plot_simulation(
            suggestions=self.suggestions,
            actions=self.actions,
            rewards=self.rewards,
            compliances=self.compliances,
            est_compliances=self.estimated_compliances,
            title_suffix=self.user_profile.get("name", "user"),
            out_path=str(plot_path),
        )
