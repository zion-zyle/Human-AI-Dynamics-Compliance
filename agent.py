# agent.py (최종 수정 버전)
import numpy as np
import os
from datetime import datetime

class Agent:
    # ... [ __init__ 및 다른 메서드들은 이전 답변과 동일 ] ...
    def __init__(
        self,
        action_space,
        goal_behavior=4.0,
        reward_weight_compliance=1.0,
        reward_weight_goal=1.5,
        penalty_weight_suggestion=0.1,
        softness_mu=0.1,
        softness_goal=0.1,
        lr_q=0.6,
        lr_compliance=0.6,
        lr_mu=0.6,
        policy_temp_decay=0.04,
        user_age=None,
        user_gender=None,
        model_name="gpt-5-nano",
    ):
        self.action_space = np.array(action_space)
        self.num_actions = len(action_space)
        self.q_values = np.zeros(self.num_actions)
        self.goal_behavior = goal_behavior
        self.estimated_compliance = 0.5
        self.estimated_behavior_mean = goal_behavior
        self.reward_weight_compliance = reward_weight_compliance
        self.reward_weight_goal = reward_weight_goal
        self.penalty_weight_suggestion = penalty_weight_suggestion
        self.penalty_weight_behavior_mean = 0.1
        self.learning_rate_q = lr_q
        self.learning_rate_compliance = lr_compliance
        self.learning_rate_behavior_mean = lr_mu
        self.policy_temperature = 1.0
        self.min_policy_temperature = 0.1
        self.policy_temp_decay = policy_temp_decay
        self.softness_mu = softness_mu
        self.softness_goal = softness_goal
        self.action_history = []
        self.prev_suggestion_idx = None
        self.user_profile = {"age": user_age, "gender": user_gender}
        self.inferred_user_profile = {}
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY") 
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.run_context = {}

    def _update_behavior_mean(self, action):
        if action is None: return
        self.action_history.append(action)
        window = 10
        if len(self.action_history) >= window:
            self.estimated_behavior_mean = np.mean(self.action_history[-window:])
        else:
            self.estimated_behavior_mean += self.learning_rate_behavior_mean * (action - self.estimated_behavior_mean)
        self.estimated_behavior_mean = np.clip(self.estimated_behavior_mean, 0.0, 5.0)

    def _update_compliance(self, actual_compliance):
        if actual_compliance is None: return
        error = actual_compliance - self.estimated_compliance
        self.estimated_compliance += self.learning_rate_compliance * error
        self.estimated_compliance = np.clip(self.estimated_compliance, 0.0, 1.0)

    def _update_q(self, suggestion_idx, reward):
        if reward is None: return
        old_q = self.q_values[suggestion_idx]
        self.q_values[suggestion_idx] += self.learning_rate_q * (reward - old_q)

    def _decay_temperature(self):
        mu_distance = abs(self.estimated_behavior_mean - self.goal_behavior)
        max_distance = np.ptp(self.action_space)
        ratio = np.clip(mu_distance / max_distance, 0.0, 1.0)
        target_temp = self.min_policy_temperature + ratio * (1.0 - self.min_policy_temperature)
        self.policy_temperature = (1 - self.policy_temp_decay) * self.policy_temperature + self.policy_temp_decay * target_temp

    def reward(self, suggestion_idx, action, actual_compliance):
        if action is None or actual_compliance is None:
            return 0.0, 0.0
        self._update_compliance(actual_compliance)
        self._update_behavior_mean(action)
        goal_score = np.exp(-((action - self.goal_behavior) ** 2))
        mu_score = np.exp(-((self.estimated_behavior_mean - self.goal_behavior) ** 2))
        suggestion_score = np.exp(-((self.action_space[suggestion_idx] - self.goal_behavior) ** 2))
        total_reward = (
            self.reward_weight_compliance * actual_compliance +
            self.reward_weight_goal * goal_score +
            self.penalty_weight_behavior_mean * mu_score +
            self.penalty_weight_suggestion * suggestion_score
        )
        self._update_q(suggestion_idx, total_reward)
        return total_reward, actual_compliance

    def policy(self):
        scaled_q = self.q_values / self.policy_temperature
        idx_mu = np.argmin(np.abs(self.action_space - self.estimated_behavior_mean))
        idx_goal = np.argmin(np.abs(self.action_space - self.goal_behavior))
        weight_mu = np.exp(-self.softness_mu * np.abs(np.arange(len(self.action_space)) - idx_mu))
        weight_goal = np.exp(-self.softness_goal * np.abs(np.arange(len(self.action_space)) - idx_goal))
        smooth_weight = weight_mu * weight_goal
        adjusted_scaled = scaled_q + np.log(smooth_weight + 1e-8)
        probs = np.exp(adjusted_scaled - np.max(adjusted_scaled))
        probs /= np.sum(probs)
        suggestion_idx = np.random.choice(len(self.action_space), p=probs)
        self.prev_suggestion_idx = suggestion_idx
        suggestion = self.action_space[suggestion_idx]
        self._decay_temperature()
        return suggestion, suggestion_idx, probs

    def format_agent_1st_session_prompt(self) -> str:
        return """
        You are a behavior coaching agent starting your first session with a new user.

        ---
        ## 📝 Your Core Directives:
        1.  **Review the user's previous statements before asking a question.** Do NOT ask for information the user has already provided.
        2.  Your primary goal is to gather information on all user attributes listed below.
        3.  Ask **only one question at a time**, targeting a single attribute.
        4.  Once you believe an attribute has been answered, move on to the next UNKNOWN attribute.
        5.  After gathering most attributes, you must collaboratively set a `goal_behavior` with the user.
        ---
        
        ## 🧭 Attributes to Investigate:
        - `condition`: Known health conditions that influence diet.
        - `mu`: Regularity of current eating behaviors.
        - `beta`: Sensitivity to external suggestions (e.g., trends, social cues).
        - `alpha`: Flexibility in adopting new habits.
        - `gamma`: Sensitivity to emotional/environmental factors.
        - `memory`: Ability to recall and reflect on eating patterns.
        - `delta`: Need for structure/stability in dietary change.
        - `epsilon`: Tendency for spontaneous or irregular eating behaviors.
        - `goal_behavior`: A desired dietary behavior level from 1.0 (very poor) to 5.0 (highly healthy).

        ## ➡️ Current Conversation History:
        (The conversation history will be supplied by the system here.)

        ---
        ### Output Format (JSON):
        You MUST output your response in JSON format.

        ```json
        {
            "monologue": "My last question was about [previous attribute]. The user's response indicates their [attribute] is [inferred value]. Now, I will ask about the next unknown attribute: [next attribute].",
            "utterance": "Your next question to the user.",
            "endkey": false
        }
        ```
        """.strip()
    
    def format_agent_1st_session_analysis_prompt(self) -> str:
        return """
        You are analyzing the user's responses from a first-time dietary behavior coaching session.
        Based on the entire conversation, populate the `inferred_attributes` object. If any attribute is not clearly inferable, leave it as null.
        
        ## ➡️ Full Conversation Log:
        (The full conversation log is provided by the system.)
        
        ### User Profile Attributes:
        - `condition`, `mu`, `beta`, `alpha`, `gamma`, `memory`, `delta`, `epsilon`

        ### User Profile Output Format:
        ```json
        {
            "goal_behavior": 4.0,
            "inferred_attributes": {
                "condition": "e.g., Type 2 Diabetes",
                "mu": "e.g., Highly irregular",
                "beta": "e.g., Moderately suggestible",
                "alpha": "e.g., Flexible and open to change",
                "gamma": "e.g., Highly emotionally influenced",
                "memory": "e.g., Poor short-term recall",
                "delta": "e.g., Requires clear structure",
                "epsilon": "e.g., Often deviates from plans"
            }
        }
        ```
        """.strip()

    def format_agent_prompt(self, suggestion_score, suggestion_history, prior_analysis, planned_suggestion) -> str:
        ctx = self.run_context or {}
        comp = ctx.get("compliance_summary", {}) or {}
        now = datetime.now()
        current_day = now.strftime("%A")
        current_time = now.strftime("%I:%M %p")
        weather_condition = "Clear" # Placeholder
        temperature = "32°C" # Placeholder

        def _d(v, fallback="Unknown"):
            return fallback if v is None else v

        return f"""
        You are a **dietary behavior coaching agent** working with a user in an ongoing session.
        ---
        ## 🌍 Contextual Factors
        - **Location**: Suwon-si, South Korea
        - **Current Day**: {current_day}
        - **Current Time**: {current_time}
        - **Current Weather**: {weather_condition}, {temperature}
        
        **Instruction**: Consider these environmental factors. For example, a stressful Monday or rainy weather might influence the user's mood. Tailor your tone and suggestion accordingly.
        ---
        ## 📈 Compliance (so far)
        - Agent running estimate: {_d(comp.get('estimated_by_agent'), 'NA')}
        - Mean (all sessions): {_d(comp.get('mean'), 'NA')}
        ---
        ## 🔢 Current Behavior Snapshot
        - **Planned Numeric Suggestion (internal)**: {planned_suggestion:.2f}
        - **Recent Suggestion History**: {suggestion_history}
        - **Prior Agent Analysis**: {prior_analysis}
        ---
        ## 🧠 User Behavioral Profile
        - **Condition**: {self.inferred_user_profile.get('condition', 'Unknown')}
        - **Emotional Sensitivity (γ)**: {self.inferred_user_profile.get('gamma', 'Unknown')}
        - **Habit Adaptability (α)**: {self.inferred_user_profile.get('alpha', 'Unknown')}
        ---
        ## 🎯 Instructions:
        1.  **Reflect** on all information: the user's profile, past compliance, and current context.
        2.  **Generate a brief, supportive, and actionable suggestion** aligned with the planned numeric suggestion.
        3.  Include an **internal monologue** explaining your reasoning, explicitly mentioning how contextual factors influenced your choice.

        ### Output Format (JSON):
        {{
            "monologue": "Based on the hot weather and it being a Friday afternoon, the user might be tired. I will propose a simple, low-effort suggestion. This aligns with their emotional sensitivity.",
            "utterance": "Your warm, motivational suggestion to the user.",
            "endkey": false
        }}
        """.strip()

    def format_agent_session_analysis_prompt(self, last_suggestion: float) -> str:
        return f"""
        You are a behavioral psychologist analyzing a conversation between an AI coach and a user. 
        Your task is to **predict the user's most likely action** based on their personality, tone, and stated intentions.

        **CRITICAL INSTRUCTION:** The user often says "I will try..." or "Okay, I'll do that." Do **NOT** interpret this as inaction. If the user's tone is positive and they agree to the suggestion, you must infer that they **performed an action that is highly aligned with the suggestion.**

        ### Case Examples for Inference:
        - User says "Wow, that's a great idea! I will definitely try that today." -> **Estimate a high action value (e.g., 4.0-5.0).**
        - User says "Hmm, I'm not sure, but I guess I can try." -> **Estimate a moderate action value (e.g., 2.5-3.5).**
        - User says "No, I don't think that will work for me." -> **Estimate a low action value (e.g., 1.0-2.0).**
        ---
        ### 💬 Conversation to Analyze:
        (The full conversation log is provided by the system.)
        ---
        ### 📝 Your Tasks:
        1.  **Predict User's Action (`user_action_estimate`)**: Based on their final statements and overall tone, what is the most probable action (1.0 to 5.0) the user took?
        2.  **Estimate Compliance (`compliance_estimate`)**: Calculate compliance based on your predicted action and the agent's last suggestion of **S = {last_suggestion:.2f}**. (Formula: 1 - |action_estimate - S| / 4.0)
        3.  **Provide Psychological Analysis**: Briefly analyze the user's cognitive state and provide coaching notes.
        ---
        ### 🧾 Output Format (JSON):
        ```json
        {{
            "user_action_estimate": 4.5,
            "compliance_estimate": 0.89,
            "confidence": 0.85,
            "basis": "The user responded with strong enthusiasm and commitment ('Sounds good. I’ll try...'), making it highly probable they followed the suggestion closely.",
            "cognitive_dissonance": "Concise summary of user's internal conflicts.",
            "negative_thought_patterns": "Recurring negative beliefs.",
            "emotional_triggers": "Factors driving unhealthy eating.",
            "effective_reinforcement": ["Empathy", "Praise"],
            "coaching_notes": "The user responds well to clear, actionable plans. Continue providing concrete examples."
        }}
        ```
        """.strip()