import os
import math
from typing import Dict, Optional, List, Tuple, Union
import numpy as np

# 옵션 리스트(기존과 동일)
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
    respond()가 상세 로깅을 위해 return_details=True일 때 각종 내부값을 함께 반환합니다.
    """
    def __init__(self, user_profile: Dict[str, str], model_name: str = "gpt-5-nano"):
        self.user_profile = user_profile
        self._initialize_llm_params(model_name)
        self._initialize_numeric_params()
        self._apply_preset_overrides()   # ← 프리셋(이름/옵션)에 따른 개성 강화

    # ---------- LLM 세팅 ----------
    def _initialize_llm_params(self, model_name: str):
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    # ---------- 수치 파라미터 초기화(기본 스케일) ----------
    def _initialize_numeric_params(self):
        # 기본 맵핑
        self.behavior_mean = self._map_text_to_value(MU_LIST, self.user_profile['mu'], 1.0, 5.0)

        # 순응감도(β) 살짝 완화 → 제안에 더 잘 반응 (기본값, 이후 프리셋에서 가감 예정)
        base_beta = self._map_text_to_value(BETA_LIST, self.user_profile['beta'], 0.1, 8.0, reverse=True)
        self.compliance_sensitivity = 0.7 * base_beta  # 기본 스케일

        # 적응률(EMA형 이동)
        self.adaptation_rate = self._map_text_to_value(DELTA_LIST, self.user_profile['delta'], 0.03, 0.30, reverse=True)

        # 노이즈
        self.noise_sensitivity = 0.8 * self._map_text_to_value(EPSILON_LIST, self.user_profile['epsilon'], 0.1, 1.5)
        self.min_noise = 0.02

        # 히스토리(표본 유지용)
        self.history: List[float] = []

        # 프리셋 전용 임계값(기본 비활성)
        self.abrupt_drop_threshold = 0.0  # |S - mu| 임계치
        self.abrupt_drop_factor = 1.0     # 임계 초과 시 순응 배수

    # ---------- 텍스트 옵션 → 수치 맵핑 ----------
    def _map_text_to_value(self, option_list: List[str], current_value: str,
                           min_val: float, max_val: float, reverse: bool = False) -> float:
        try:
            index = option_list.index(current_value)
            total_items = len(option_list)
            normalized_value = index / (total_items - 1)
            if reverse:
                normalized_value = 1.0 - normalized_value
            return normalized_value * (max_val - min_val) + min_val
        except ValueError:
            return (min_val + max_val) / 2

    # ---------- 프리셋(이름/옵션)에 의한 개성 강화 ----------
    def _apply_preset_overrides(self):
        """
        user_profile["preset"] 또는 name에 포함된 키워드로 개성을 강화합니다.
        - independent: 제안 둔감, 적응 느림
        - compliant: 제안 민감, 적응 빠름, 저노이즈
        - adaptive: 중간값, 순응↑일수록 빠른 적응
        - resistant: 큰 차이에 순응 급락, 적응 매우 느림
        - high_noise: 노이즈 큼
        """
        def clamp(x, lo, hi): return float(np.clip(float(x), lo, hi))
        preset_key = (self.user_profile.get("preset") or self.user_profile.get("name") or "").lower()

        if "independent" in preset_key:
            self.compliance_sensitivity = clamp(self.compliance_sensitivity * 2.0, 0.2, 12.0)
            self.adaptation_rate = clamp(self.adaptation_rate * 0.5, 0.01, 0.20)
            self.noise_sensitivity = clamp(self.noise_sensitivity * 1.0, 0.05, 2.0)

        elif "compliant" in preset_key:
            self.compliance_sensitivity = clamp(self.compliance_sensitivity * 0.35, 0.05, 3.0)
            self.adaptation_rate = clamp(self.adaptation_rate * 1.8, 0.05, 0.50)
            self.noise_sensitivity = clamp(self.noise_sensitivity * 0.6, 0.02, 1.0)
            self.min_noise = 0.01

        elif "adaptive" in preset_key:
            self.compliance_sensitivity = clamp(self.compliance_sensitivity * 0.8, 0.1, 6.0)
            self.adaptation_rate = clamp(self.adaptation_rate * 1.2, 0.03, 0.40)
            self.noise_sensitivity = clamp(self.noise_sensitivity * 0.9, 0.03, 1.5)

        elif "resistant" in preset_key:
            self.compliance_sensitivity = clamp(self.compliance_sensitivity * 2.5, 0.5, 12.0)
            self.adaptation_rate = clamp(self.adaptation_rate * 0.5, 0.01, 0.15)
            self.noise_sensitivity = clamp(self.noise_sensitivity * 0.85, 0.03, 1.2)
            self.abrupt_drop_threshold = 0.6   # |S - mu|가 이보다 크면
            self.abrupt_drop_factor = 0.5      # 순응 50% 급락

        elif "high_noise" in preset_key:
            self.noise_sensitivity = clamp(self.noise_sensitivity * 1.8, 0.2, 3.0)
            self.min_noise = 0.04
            self.adaptation_rate = clamp(self.adaptation_rate * 0.9, 0.02, 0.30)
            self.compliance_sensitivity = clamp(self.compliance_sensitivity * 1.1, 0.1, 8.0)

        # 그 외 프리셋 미지정: 기본 스케일 유지(변경 없음)

    # ---------- 순응 확률 ----------
    def compliance_prob(self, suggestion: float) -> float:
        # 기본: 가우시안 형태
        base = float(np.exp(-self.compliance_sensitivity * (suggestion - self.behavior_mean) ** 2))
        # 큰 차이(임계 초과) 급락(주로 resistant에서 활성)
        if self.abrupt_drop_threshold > 0.0 and abs(suggestion - self.behavior_mean) > self.abrupt_drop_threshold:
            base *= self.abrupt_drop_factor
        return float(np.clip(base, 0.0, 1.0))

    # ---------- 행동 생성 ----------
    def respond(self, suggestion: float, return_details: bool = False) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        사용자 행동 샘플을 생성하고, 요청 시 내부값 세트를 함께 반환합니다.
        action = clip((1 - c)*mu + c*suggestion + noise, 1, 5)
        """
        mean_before = float(self.behavior_mean)
        compliance = self.compliance_prob(suggestion)

        noise_std = self.noise_sensitivity * (1 - compliance) + self.min_noise
        noise = float(np.random.normal(0.0, noise_std))

        action = float(np.clip((1 - compliance) * self.behavior_mean + compliance * suggestion + noise, 1.0, 5.0))

        # 히스토리 유지
        self.history.append(action)
        if len(self.history) > 200:
            self.history.pop(0)

        # EMA형 적응: 순응이 높을수록 평균을 더 빨리 이동
        adapt_gain = self.adaptation_rate * (0.5 + 0.5 * compliance)  # 0.5~1.0 배
        self.behavior_mean += adapt_gain * (action - self.behavior_mean)
        self.behavior_mean = float(np.clip(self.behavior_mean, 1.0, 5.0))

        if not return_details:
            return action

        details = {
            "compliance": float(compliance),
            "behavior_mean_before": mean_before,
            "behavior_mean_after": float(self.behavior_mean),
            "suggestion": float(suggestion),
            "noise": float(noise),
            "noise_std": float(noise_std)
        }
        return action, details
