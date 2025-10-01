import numpy as np

# 1. user.py에서 리스트 정의와 변환 함수를 그대로 가져옵니다.
# -----------------------------------------------------------------
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
# ... (ALPHA, DELTA, EPSILON 리스트도 여기에 복사) ...
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
    "Behavaves predictably with almost no deviations", "Rarely shows exceptions to routine",
    "Occasional deviation from typical patterns", "Frequently exhibits irregular behaviors",
    "Consistently unpredictable and erratic"
]


def _map_text_to_value(option_list: list, current_value: str, min_val: float, max_val: float, reverse: bool = False) -> float:
    try:
        index = option_list.index(current_value)
        total_items = len(option_list)
        normalized_value = index / (total_items - 1)
        if reverse: normalized_value = 1.0 - normalized_value
        return normalized_value * (max_val - min_val) + min_val
    except ValueError: return (min_val + max_val) / 2
# -----------------------------------------------------------------


# 2. user.py의 _initialize_numeric_params에 정의된 목표 범위를 설정합니다.
# -----------------------------------------------------------------
traits = {
    "μ (behavior_mean)": {
        "list": MU_LIST, "range": (1.0, 5.0), "reverse": False
    },
    "β (compliance_sensitivity)": {
        "list": BETA_LIST, "range": (0.1, 8.0), "reverse": True
    },
    "α (inertia)": {
        "list": ALPHA_LIST, "range": (0.0, 0.9), "reverse": True
    },
    "δ (adaptation_rate)": {
        "list": DELTA_LIST, "range": (0.01, 0.2), "reverse": True
    },
    "ε (noise_sensitivity)": {
        "list": EPSILON_LIST, "range": (0.1, 1.5), "reverse": False
    }
}
# -----------------------------------------------------------------


# 3. 각 리스트의 변환 결과를 출력합니다.
# -----------------------------------------------------------------
for name, data in traits.items():
    print(f"--- {name} ---")
    for text in data["list"]:
        value = _map_text_to_value(data["list"], text, data["range"][0], data["range"][1], data["reverse"])
        print(f"'{text}'  =>  {value:.4f}")
    print("-" * (len(name) + 6) + "\n")
# -----------------------------------------------------------------