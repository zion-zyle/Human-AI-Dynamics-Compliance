# user.py
import os
import math
from typing import Dict, Optional, List
import numpy as np

# 'main.py'를 수정하지 않기 위해 프로필 리스트들을 내장합니다.
MU_LIST = [
    "Highly irregular eating patterns", "Somewhat irregular eating habits",
    "Moderately regular dietary routine", "Slightly structured meal schedule",
    "Strictly consistent eating habits"
]
BETA_LIST = [
    "Highly resistant to dietary suggestions", "Somewhat resistant to behavioral influence",
    "Moderately compliant with guidance", "Easily influenced by suggestions",
    "Highly suggestible and reactive to guidance"
]
ALPHA_LIST = [
    "Extremely resistant to behavioral change", "Rarely adopts new eating behaviors",
    "Occasionally adapts eating habits", "Frequently adopts suggested behaviors",
    "Immediately responsive to new habits"
]
DELTA_LIST = [
    "Highly reactive to small pattern changes", "Adapts with minimal stability required",
    "Moderately stable before behavior change", "Requires significant stability to change",
    "Changes only after long-term behavioral reinforcement"
]
EPSILON_LIST = [
    "Behaves predictably with almost no deviations", "Rarely shows exceptions to routine",
    "Occasional deviation from typical patterns", "Frequently exhibits irregular behaviors",
    "Consistently unpredictable and erratic"
]
MEMORY_LIST = [
    "Poor recall of recent eating behaviors", "Able to recall patterns for about 1 week",
    "Able to recall for approximately 2 weeks", "Able to maintain pattern memory over 1 month",
    "Long-term retention of dietary routines"
]

class UserLlm:
    """
    발화(utterance)는 LLM으로, 행동(action)은 수학 함수로 결정하는 하이브리드 사용자 클래스.
    """
    def __init__(self, user_profile: Dict[str, str], model_name: str = "gpt-5-nano"):
        self.user_profile = user_profile
        self._initialize_llm_params(model_name)
        self._initialize_numeric_params()

    def _initialize_llm_params(self, model_name: str):
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.headers = { "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}" }

    def _initialize_numeric_params(self):
        self.behavior_mean = self._map_text_to_value(MU_LIST, self.user_profile['mu'], 1.0, 5.0)
        self.compliance_sensitivity = self._map_text_to_value(BETA_LIST, self.user_profile['beta'], 0.1, 8.0, reverse=True)
        self.adaptation_rate = self._map_text_to_value(DELTA_LIST, self.user_profile['delta'], 0.01, 0.2, reverse=True)
        self.noise_sensitivity = self._map_text_to_value(EPSILON_LIST, self.user_profile['epsilon'], 0.1, 1.5)
        try:
            mem_index = MEMORY_LIST.index(self.user_profile['memory'])
            self.memory = (mem_index + 1) * 7
        except ValueError: self.memory = 14
        self.min_noise = 0.05
        self.convergence_threshold = 0.1
        self.history: List[float] = []

    def _map_text_to_value(self, option_list: List[str], current_value: str, min_val: float, max_val: float, reverse: bool = False) -> float:
        try:
            index = option_list.index(current_value)
            total_items = len(option_list)
            normalized_value = index / (total_items - 1)
            if reverse: normalized_value = 1.0 - normalized_value
            return normalized_value * (max_val - min_val) + min_val
        except ValueError: return (min_val + max_val) / 2

    # --- ▼▼▼ 행동 결정을 위한 수학 함수들 ▼▼▼ ---
    def compliance_prob(self, suggestion: float) -> float:
        return np.exp(-self.compliance_sensitivity * (suggestion - self.behavior_mean) ** 2)

    def respond(self, suggestion: float) -> float:
        compliance = self.compliance_prob(suggestion)
        noise_std = self.noise_sensitivity * (1 - compliance) + self.min_noise
        noise = np.random.normal(0, noise_std)
        action = float(np.clip((1 - compliance) * self.behavior_mean + compliance * suggestion + noise, 1.0, 5.0))
        self.history.append(action)
        if len(self.history) > self.memory: self.history.pop(0)
        if len(self.history) == self.memory and np.std(self.history) < self.convergence_threshold:
            self.behavior_mean += self.adaptation_rate * np.sign(np.mean(self.history) - self.behavior_mean)
        return action

    # --- 발화 생성을 위한 LLM 프롬프트 함수 (기존과 동일) ---
    def _tail_str(self, seq: Optional[List[float]], n: int = 5) -> str:
        if not seq: return "None"
        nums = [float(x) for x in seq if isinstance(x, (int, float)) and not math.isnan(x)]
        return ", ".join(f"{v:.2f}" for v in nums[-n:]) if nums[-n:] else "None"

    def format_user_prompt(
        self,
        recommendation_history: Optional[List[float]] = None,
        action_history: Optional[List[float]] = None,
        history_window: int = 5
    ) -> str:
        rec_tail = self._tail_str(recommendation_history, history_window)
        act_tail = self._tail_str(action_history, history_window)
        return f"""
        ## 🧑‍⚕️ Dietary Coaching User Profile
        You are a user receiving coaching from an AI to improve your dietary habits.
        - **Your Age**: {self.user_profile['age']}
        - **Your Gender**: {self.user_profile['gender']}
        - **Your Condition**: {self.user_profile['condition']}
        - **Your typical habits (μ)** are described as: {self.user_profile['mu']}
        - **You tend to be (β)**: {self.user_profile['beta']}
        - **You are (γ)**: {self.user_profile['gamma']} to external influences.
        ### 📈 Recent Context
        - Recent agent recommendations (last {history_window}): [{rec_tail}]
        - My actual actions (last {history_window}): [{act_tail}]
        ### 🎯 Your Task
        React to the agent's latest suggestion in a natural, conversational way that is consistent with your profile. Your response should be what you would say out loud.
        ### ✏️ Output Instructions
        Generate your response in JSON format with only the 'utterance' key.
        \```json
        {{
            "utterance": "What you would say out loud to the agent."
        }}
        \```
        """.strip()