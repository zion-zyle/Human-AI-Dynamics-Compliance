# === main.py (병렬 배치 실행: 5명씩, 사람별 폴더 저장) ===
import os
from dotenv import load_dotenv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from user import UserLlm
from agent import Agent
from simulator import Simulator
from visualization import plot_simulation

# 5명씩 병렬 처리
MAX_WORKERS = 5


def run_one(user_profile: dict, global_idx: int) -> str:
    """
    한 명의 시뮬레이션을 실행하고, 결과를 folder{global_idx+1}/ 하위에 저장.
    Simulator는 base_dir을 받아 io_logs/, sessions/, simulation_log.txt를 해당 폴더 아래에 기록.
    """
    base_dir = f"folder{global_idx+1}"
    os.makedirs(base_dir, exist_ok=True)

    action_space = np.linspace(0.0, 5.0, 100)

    user = UserLlm(user_profile, model_name="gpt-5-nano")
    agent = Agent(
        action_space=action_space,
        user_age=user_profile.get("age"),
        user_gender=user_profile.get("gender"),
        model_name="gpt-5-nano",
    )

    sim = Simulator(
        user=user,
        agent=agent,
        action_space=action_space,
        total_steps=56,     # 기존과 동일
        base_dir=base_dir,  # 사람별 폴더 분리 저장
    )

    sim.train()

    # 플롯도 같은 폴더 하위에 저장 (plots/NAME.png)
    plot_simulation(
        sim,
        save=True,
        filename=user_profile.get("name", f"user_{global_idx+1}"),
        base_dir=base_dir,
    )

    return os.path.abspath(base_dir)


def chunked_indices(n_items: int, size: int):
    """(start, end) 구간을 yield. end는 exclusive."""
    for start in range(0, n_items, size):
        yield start, min(start + size, n_items)


if __name__ == "__main__":
    # --- 옵션 리스트 (네 기존 main.py와 동일한 인덱스 사용 가능) ---
    age_list = [
        "Teenager (10s)",
        "Young adult (20s)",
        "Adult (30s)",
        "Middle-aged (40s)",
        "Older adult (50+)"
    ]

    gender_list = [
        "Male",
        "Female",
        "Non-binary / Other",
        "Prefer not to say"
    ]

    condition_list = [
        "None",
        "Overeating (Hyperphagia)",
        "Binge eating disorder (BED)",
        "Anorexia nervosa",
        "Night eating syndrome",
        "Glycemic regulation issues",
        "Gastrointestinal disorders",
        "Other"
    ]

    mu_list = [
        "Highly irregular eating patterns",
        "Somewhat irregular eating habits",
        "Moderately regular dietary routine",
        "Slightly structured meal schedule",
        "Strictly consistent eating habits"
    ]
    beta_list = [
        "Highly resistant to dietary suggestions",
        "Somewhat resistant to behavioral influence",
        "Moderately compliant with guidance",
        "Easily influenced by suggestions",
        "Highly suggestible and reactive to guidance"
    ]
    alpha_list = [
        "Extremely resistant to behavioral change",
        "Rarely adopts new eating behaviors",
        "Occasionally adapts eating habits",
        "Frequently adopts suggested behaviors",
        "Immediately responsive to new habits"
    ]
    gamma_list = [
        "Insensitive to emotional or environmental stimuli",
        "Slightly responsive to contextual cues",
        "Moderately sensitive to external changes",
        "Highly influenced by situational factors",
        "Extremely vulnerable to emotional or environmental triggers"
    ]
    memory_list = [
        "Poor recall of recent eating behaviors",
        "Able to recall patterns for about 1 week",
        "Able to recall for approximately 2 weeks",
        "Able to maintain pattern memory over 1 month",
        "Long-term retention of dietary routines"
    ]
    delta_list = [
        "Highly reactive to small pattern changes",
        "Adapts with minimal stability required",
        "Moderately stable before behavior change",
        "Requires significant stability to change",
        "Changes only after long-term behavioral reinforcement"
    ]
    epsilon_list = [
        "Behaves predictably with almost no deviations",
        "Rarely shows exceptions to routine",
        "Occasional deviation from typical patterns",
        "Frequently exhibits irregular behaviors",
        "Consistently unpredictable and erratic"
    ]

    # --- 여기서부터 네가 원하는 프로필을 계속 추가/수정하면 됨 ---
    user_profiles = [
        { 
            "name": "independent_user", 
            "age": age_list[2],          # 30대 
            "gender": gender_list[1],    # 여성 
            "condition": condition_list[4],  # 야식 
            "mu": mu_list[2],            # 보통 
            "beta": beta_list[0],        # 전혀 영향받지 않음 
            "alpha": alpha_list[0],      # 절대 습관을 바꾸지 않음 
            "gamma": gamma_list[0],      # 잡식적이며 환경에 둔감함 
            "memory": memory_list[2],    # 2주 정도 기억 
            "delta": delta_list[4],      # 아주 안정적이어야 바뀜 
            "epsilon": epsilon_list[4]   # 항상 예외적 행동 
        }, 
        { 
            "name": "compliant_user", 
            "age": age_list[1],          # 20대 
            "gender": gender_list[0],    # 남성 
            "condition": condition_list[1],  # 과식 
            "mu": mu_list[3],            # 조금 규칙적임 
            "beta": beta_list[4],        # 매우 민감하게 반응 
            "alpha": alpha_list[4],      # 즉시 반응함 
            "gamma": gamma_list[4],      # 환경 변화에 매우 취약함 
            "memory": memory_list[4],    # 장기 기억함 
            "delta": delta_list[1],      # 조금만 바뀌어도 적응 
            "epsilon": epsilon_list[1]   # 예외가 거의 없음 
        }, 
        {
            "name": "adaptive_user", 
            "age": age_list[2],          # 30대
            "gender": gender_list[1],    # 여성
            "condition": condition_list[2],  # 폭식
            "mu": mu_list[1],            # 보통
            "beta": beta_list[2],        # 보통
            "alpha": alpha_list[2],      # 가끔 바꿈
            "gamma": gamma_list[2],      # 보통
            "memory": memory_list[3],    # 1달 이상 기억
            "delta": delta_list[2],      # 보통
            "epsilon": epsilon_list[2]   # 가끔 예외 발생
        },
        { 
            "name": "high_noise_user", 
            "age": age_list[3],          # 40대 
            "gender": gender_list[0],    # 남성 
            "condition": condition_list[5],  # 당 조절 문제 
            "mu": mu_list[1],            # 조금 불규칙함 
            "beta": beta_list[2],        # 보통 
            "alpha": alpha_list[1],      # 거의 바꾸지 않음 
            "gamma": gamma_list[4],      # 환경 변화에 매우 취약함 
            "memory": memory_list[1],    # 일주일 정도 기억 
            "delta": delta_list[3],      # 웬만해선 안 바뀜 
            "epsilon": epsilon_list[3]   # 예외가 많음 
        }, 
        { 
            "name": "resistant_user", 
            "age": age_list[4],          # 50대 이상 
            "gender": gender_list[1],    # 여성 
            "condition": condition_list[3],  # 거식 
            "mu": mu_list[0],            # 매우 불규칙함 
            "beta": beta_list[0],        
            "alpha": alpha_list[1],      # 거의 바꾸지 않음 
            "gamma": gamma_list[3],      # 쉽게 흔들림 
            "memory": memory_list[0],    # 거의 기억 못함 
            "delta": delta_list[3],      # 웬만해선 안 바뀜 
            "epsilon": epsilon_list[3]   # 예외가 많음 
        },
              # <-- 계속해서 여기에 dict를 추가하면 됨
    ]

    # --- 실행 ---
    load_dotenv()

    created_dirs = []

    # 5명씩 끊어서 병렬 실행
    for start, end in chunked_indices(len(user_profiles), MAX_WORKERS):
        group = user_profiles[start:end]
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            # 전역 인덱스(start 기준)를 써서 folder1..folderN 겹치지 않게
            futures = [
                ex.submit(run_one, profile, global_idx)
                for global_idx, profile in enumerate(group, start=start)
            ]
            for fut in as_completed(futures):
                created_dirs.append(fut.result())

    print("\nCreated result folders:")
    for d in sorted(set(created_dirs)):
        print(" -", d)
