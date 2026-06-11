import gymnasium as gym
import hydra
import numpy as np
from omegaconf import DictConfig

from rl_exercises.week_4.dqn import DQNAgent
from rl_exercises.week_5.policy_gradient import set_seed
from rnd_ppo import RNDPPOAgent
from rnd_dqn import RNDDQNAgent

from matplotlib import pyplot as plt  # noqa: F401


def train_rnd_dqn(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 3) TODO: instantiate & train the agent
    agent = RNDDQNAgent(
        env,
        buffer_capacity=cfg.dqn_rnd.agent.buffer_capacity,
        batch_size=cfg.dqn_rnd.agent.batch_size,
        lr=cfg.dqn_rnd.agent.learning_rate,
        gamma=cfg.dqn_rnd.agent.gamma,
        epsilon_start=cfg.dqn_rnd.agent.epsilon_start,
        epsilon_final=cfg.dqn_rnd.agent.epsilon_final,
        epsilon_decay=cfg.dqn_rnd.agent.epsilon_decay,
        target_update_freq=cfg.dqn_rnd.agent.target_update_freq,
        seed=cfg.seed,
        rnd_hidden_size=cfg.dqn_rnd.rnd.hidden_size,
        rnd_lr=cfg.dqn_rnd.rnd.learning_rate,
        rnd_update_freq=cfg.dqn_rnd.rnd.update_freq,
        rnd_n_layers=cfg.dqn_rnd.rnd.n_layers,
        rnd_reward_weight=cfg.dqn_rnd.rnd.reward_weight,
    )

    agent.train(cfg.dqn_rnd.train.num_frames, cfg.dqn_rnd.train.eval_interval)

    env.reset()

    return agent, env


def train_eq_dqn(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 2) map config to agent kwargs
    agent_kwargs = dict(
        buffer_capacity=cfg.dqn_eq.agent.buffer_capacity,
        batch_size=cfg.dqn_eq.agent.batch_size,
        lr=cfg.dqn_eq.agent.learning_rate,
        gamma=cfg.dqn_eq.agent.gamma,
        epsilon_start=cfg.dqn_eq.agent.epsilon_start,
        epsilon_final=cfg.dqn_eq.agent.epsilon_final,
        epsilon_decay=cfg.dqn_eq.agent.epsilon_decay,
        target_update_freq=cfg.dqn_eq.agent.target_update_freq,
        seed=cfg.seed,
        hidden_dim=cfg.dqn_eq.agent.get("hidden_dim", 64),
        hidden_layers=cfg.dqn_eq.agent.get("hidden_layers", 2),
        double_dqn=cfg.dqn_eq.agent.get("double_dqn", False),
        prioritized_replay=cfg.dqn_eq.agent.get("prioritized_replay", False),
        per_alpha=cfg.dqn_eq.agent.get("per_alpha", 0.6),
        per_beta_start=cfg.dqn_eq.agent.get("per_beta_start", 0.4),
        per_beta_frames=cfg.dqn_eq.agent.get("per_beta_frames", 10000),
        per_epsilon=cfg.dqn_eq.agent.get("per_epsilon", 1e-6),
    )

    # 3) instantiate and train
    agent = DQNAgent(env, **agent_kwargs)

    agent.train(cfg.dqn_eq.train.num_frames, cfg.dqn_eq.train.eval_interval)

    env.reset()

    return agent, env


def train_rnd_ppo(cfg: DictConfig):
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)
    agent = RNDPPOAgent(
        env,
        lr_actor=cfg.ppo_rnd.agent.lr_actor,
        lr_critic=cfg.ppo_rnd.agent.lr_critic,
        gamma=cfg.ppo_rnd.agent.gamma,
        gae_lambda=cfg.ppo_rnd.agent.gae_lambda,
        clip_eps=cfg.ppo_rnd.agent.clip_eps,
        epochs=cfg.ppo_rnd.agent.epochs,
        batch_size=cfg.ppo_rnd.agent.batch_size,
        ent_coef=cfg.ppo_rnd.agent.ent_coef,
        vf_coef=cfg.ppo_rnd.agent.vf_coef,
        seed=cfg.seed,
        hidden_size=cfg.ppo_rnd.agent.hidden_size,
        rnd_hidden_size=cfg.ppo_rnd.rnd.hidden_size,
        rnd_update_freq=cfg.ppo_rnd.rnd.update_freq,
        rnd_n_layers=cfg.ppo_rnd.rnd.n_layers,
        rnd_reward_weight=cfg.ppo_rnd.rnd.reward_weight,
    )
    agent.train(
        cfg.ppo_rnd.train.total_steps,
        cfg.ppo_rnd.train.eval_interval,
        cfg.ppo_rnd.train.eval_episodes,
    )

    env.reset()

    return agent, env


def evaluate_dqn_agent(agent: RNDDQNAgent, eval_env: gym.Env,
                       num_episodes: int = 100) -> list[float]:
    returns = []
    for _ in range(num_episodes):
        state, _ = eval_env.reset(seed=agent.seed)
        done = False
        total_r = 0.0
        while not done:
            action = agent.predict_action(state)
            state, r, term, trunc, _ = eval_env.step(action)
            done = term or trunc
            total_r += r
        returns.append(total_r)
    return returns


def evaluate_ppo_agent(agent: RNDPPOAgent, eval_env: gym.Env,
                       num_episodes: int = 100) -> list[float]:
    returns = []
    for _ in range(num_episodes):
        state, _ = eval_env.reset(seed=agent.seed)
        done = False
        total_r = 0.0
        while not done:
            action, _, _, _, _ = agent.predict_action(state)
            state, r, term, trunc, _ = eval_env.step(action)
            done = term or trunc
            total_r += r
        returns.append(total_r)
    return returns


@hydra.main(config_path="../configs/", config_name="W7L1_comp", version_base="1.1")
def main(cfg: DictConfig) -> None:
    rnd_dqn_agent, env_dqn = train_rnd_dqn(cfg)

    rnd_ppo_agent, env_ppo = train_rnd_ppo(cfg)

    eg_dqn_agent, env_eg_dqn = train_eq_dqn(cfg)

    rnd_dqn_returns = evaluate_dqn_agent(rnd_dqn_agent, env_dqn)
    rnd_ppo_returns = evaluate_ppo_agent(rnd_ppo_agent, env_ppo)
    eg_dqn_returns = evaluate_dqn_agent(eg_dqn_agent, env_eg_dqn)

    print(
        f"RND DQN Avg Return: {sum(rnd_dqn_returns) / len(rnd_dqn_returns):.2f}")
    print(
        f"RND PPO Avg Return: {sum(rnd_ppo_returns) / len(rnd_ppo_returns):.2f}")
    print(
        f"EG PPO Avg Return: {sum(eg_dqn_returns) / len(eg_dqn_returns):.2f}")

    plt.plot(np.arange(len(rnd_dqn_returns)),
             rnd_dqn_returns, label="RND DQN")
    plt.plot(np.arange(len(rnd_ppo_returns)),
             rnd_ppo_returns, label="RND PPO")
    plt.plot(np.arange(len(eg_dqn_returns)),
             eg_dqn_returns, label="EG DQN")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.title("RND DQN Training Rewards")
    plt.show()


if __name__ == "__main__":
    main()
