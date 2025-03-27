import gymnasium as gym
import numpy as np
from tqdm import tqdm
from rl2025.constants import EX2_QL_CONSTANTS as CONSTANTS
from rl2025.exercise2.agents import QLearningAgent
from rl2025.exercise2.utils import evaluate
from rl2025.util.result_processing import Run, rank_runs, get_best_saved_run


def q_learning_eval(
        env,
        config,
        q_table,
        render=False):
    """
    Evaluate configuration of Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    if render:
        eval_env = gym.make(CONFIG["env"], render_mode="human")
    else:
        eval_env = env
    return evaluate(eval_env, eval_agent, config["eval_eps_max_steps"], config["eval_episodes"])


def train(env, config):
    """
    Train and evaluate Q-Learning on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table
    """
    agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in tqdm(range(1, config["total_eps"]+1)):
        obs, _ = env.reset()
        episodic_return = 0
        t = 0

        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)
            n_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            agent.learn(obs, act, reward, n_obs, done)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = q_learning_eval(env, config, agent.q_table)
            tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table


# if __name__ == "__main__":
#     env = gym.make(CONFIG["env"])
#     total_reward, _, _, q_table = train(env, CONFIG)
import pandas as pd
import copy  # To avoid modifying CONFIG in-place

if __name__ == "__main__":
    CONFIGS = [
        {"eval_freq": 1000, "alpha": 0.05, "epsilon": 0.9, "gamma": 0.99},
        {"eval_freq": 1000, "alpha": 0.05, "epsilon": 0.9, "gamma": 0.8}
    ]

    results = []  # List to store run results

    for i in range(10):  # 10 runs per configuration
        for index, CONFIG in enumerate(CONFIGS):
            config_copy = copy.deepcopy(CONFIG)  # Avoid modifying the original CONFIG
            config_copy.update(CONSTANTS)

            env = gym.make(config_copy["env"])
            total_reward, eval_means, neg_returns, _ = train(env, config_copy)

            results.append({
                "total_reward": total_reward,
                "evaluation_return_means": eval_means.tolist() if hasattr(eval_means, "tolist") else eval_means,
                "evaluation_negative_returns": neg_returns,
                "configuration": f"config_{index}"
            })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    df_results['last_return_mean']=df_results.evaluation_return_means.apply(lambda x: x[-1])
    df_results.to_csv("results_q_learning.csv")