# main.py

import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 불러와 설정합니다.
# 이 코드는 다른 import 문보다 먼저 실행되는 것이 좋습니다.
load_dotenv()

# (선택 사항) API 키가 제대로 로드되었는지 확인하는 디버깅 코드
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✅ API 키를 성공적으로 불러왔습니다.")
else:
    print("❌ 오류: .env 파일에서 API 키를 찾지 못했거나 파일이 없습니다.")

import numpy as np
from user import UserLlm
from agent import Agent
from simulator import Simulator
from visualization import plot_simulation

if __name__ == "__main__":
    action_space = np.linspace(0.0, 5.0, 100)

    # Dietary Behavior Traits
    mu_list = [
        "Highly irregular eating patterns",
        "Somewhat irregular eating habits",
        "Moderately regular dietary routine",
        "Slightly structured meal schedule",
        "Strictly consistent eating habits"
    ]  # Baseline dietary regularity (μ)

    beta_list = [
        "Highly resistant to dietary suggestions",
        "Somewhat resistant to behavioral influence",
        "Moderately compliant with guidance",
        "Easily influenced by suggestions",
        "Highly suggestible and reactive to guidance"
    ]  # Compliance sensitivity (β)

    alpha_list = [
        "Extremely resistant to behavioral change",
        "Rarely adopts new eating behaviors",
        "Occasionally adapts eating habits",
        "Frequently adopts suggested behaviors",
        "Immediately responsive to new habits"
    ]  # Adaptability to routine (α)

    gamma_list = [
        "Insensitive to emotional or environmental stimuli",
        "Slightly responsive to contextual cues",
        "Moderately sensitive to external changes",
        "Highly influenced by situational factors",
        "Extremely vulnerable to emotional or environmental triggers"
    ]  # Sensitivity to external stimuli (γ)

    memory_list = [
        "Poor recall of recent eating behaviors",
        "Able to recall patterns for about 1 week",
        "Able to recall for approximately 2 weeks",
        "Able to maintain pattern memory over 1 month",
        "Long-term retention of dietary routines"
    ]  # Dietary memory span

    delta_list = [
        "Highly reactive to small pattern changes",
        "Adapts with minimal stability required",
        "Moderately stable before behavior change",
        "Requires significant stability to change",
        "Changes only after long-term behavioral reinforcement"
    ]  # Pattern stability threshold (δ)

    epsilon_list = [
        "Behaves predictably with almost no deviations",
        "Rarely shows exceptions to routine",
        "Occasional deviation from typical patterns",
        "Frequently exhibits irregular behaviors",
        "Consistently unpredictable and erratic"
    ]  # Likelihood of unexpected behavior (ε)

    # General Demographic Information
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

    # Clinical or Health-Related Condition
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
        # {
        #     "name": "compliant_user",
        #     "age": age_list[1],          # 20대
        #     "gender": gender_list[0],    # 남성
        #     "condition": condition_list[1],  # 과식
        #     "mu": mu_list[3],            # 조금 규칙적임
        #     "beta": beta_list[4],        # 매우 민감하게 반응
        #     "alpha": alpha_list[4],      # 즉시 반응함
        #     "gamma": gamma_list[4],      # 환경 변화에 매우 취약함
        #     "memory": memory_list[4],    # 장기 기억함
        #     "delta": delta_list[1],      # 조금만 바뀌어도 적응
        #     "epsilon": epsilon_list[1]   # 예외가 거의 없음
        # },
        #{
        #    "name": "adaptive_user", 
        #    "age": age_list[2],          # 30대
        #    "gender": gender_list[1],    # 여성
        #    "condition": condition_list[2],  # 폭식
        #    "mu": mu_list[1],            # 보통
        #    "beta": beta_list[2],        # 보통
        #    "alpha": alpha_list[2],      # 가끔 바꿈
        #    "gamma": gamma_list[2],      # 보통
        #    "memory": memory_list[3],    # 1달 이상 기억
        #    "delta": delta_list[2],      # 보통
        #    "epsilon": epsilon_list[2]   # 가끔 예외 발생
        #},
        # {
        #     "name": "high_noise_user",
        #     "age": age_list[3],          # 40대
        #     "gender": gender_list[0],    # 남성
        #     "condition": condition_list[5],  # 당 조절 문제
        #     "mu": mu_list[1],            # 조금 불규칙함
        #     "beta": beta_list[2],        # 보통
        #     "alpha": alpha_list[1],      # 거의 바꾸지 않음
        #     "gamma": gamma_list[4],      # 환경 변화에 매우 취약함
        #     "memory": memory_list[1],    # 일주일 정도 기억
        #     "delta": delta_list[3],      # 웬만해선 안 바뀜
        #     "epsilon": epsilon_list[3]   # 예외가 많음
        # },
        # {
        #     "name": "resistant_user",
        #     "age": age_list[4],          # 50대 이상
        #     "gender": gender_list[1],    # 여성
        #     "condition": condition_list[3],  # 거식
        #     "mu": mu_list[0],            # 매우 불규칙함
        #     "beta": beta_list[0],        # 
        #     "alpha": alpha_list[1],      # 거의 바꾸지 않음
        #     "gamma": gamma_list[3],      # 쉽게 흔들림
        #     "memory": memory_list[0],    # 거의 기억 못함
        #     "delta": delta_list[3],      # 웬만해선 안 바뀜
        #     "epsilon": epsilon_list[3]   # 예외가 많음
        # }
    ]

    saved_images = []

    for user_profile in user_profiles:
        print(f"\n===== Running Simulation: {user_profile['name']} =====")

        user = UserLlm(user_profile, model_name="gpt-5-nano")
        agent = Agent(action_space=action_space, user_age = user_profile["age"], user_gender = user_profile["gender"], model_name="gpt-5-nano")
        sim = Simulator(user=user, agent=agent, action_space=action_space,
                        total_steps=56)
        sim.train()
        png_path = plot_simulation(sim, save=True, filename=f"{user_profile['name']}")
        saved_images.append(png_path)

    print("\nSaved plot images:", saved_images)