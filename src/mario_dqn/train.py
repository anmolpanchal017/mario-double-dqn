from __future__ import annotations

from pathlib import Path
import shutil

from .agent import MarioAgent
from .config import ProjectConfig
from .env import make_env, seed_everything
from .logging import MetricLogger, create_run_dir, last_logged_episode


def train(
    config: ProjectConfig,
    config_path: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    seed_everything(config.run.seed)
    if resume_checkpoint is None:
        run_dir = create_run_dir(config.run.save_dir, config.run.name)
        start_episode = 0
    else:
        resume_checkpoint = Path(resume_checkpoint)
        run_dir = resume_checkpoint.parent.parent
        start_episode = last_logged_episode(run_dir) + 1
        print(f"Resuming from {resume_checkpoint}")
        print(f"Continuing run directory: {run_dir}")

    if config_path is not None and not (run_dir / "config.toml").exists():
        shutil.copy2(config_path, run_dir / "config.toml")

    env = make_env(config.env)
    logger = MetricLogger(run_dir)
    state_dim = env.observation_space
    action_dim = env.action_space.n
    agent = MarioAgent(state_dim, action_dim, config.agent, run_dir)
    if resume_checkpoint is not None:
        agent.load(resume_checkpoint)

    try:
        for episode in range(start_episode, start_episode + config.run.episodes):
            state = env.reset()
            steps = 0
            while True:
                if config.run.render:
                    env.render()
                action = agent.act(state)
                next_state, reward, done, trunc, info = env.step(action)
                terminal = done or trunc or bool(info.get("flag_get", False))
                agent.cache(state, next_state, action, reward, terminal)
                q_value, loss = agent.learn()
                logger.log_step(reward, loss, q_value)
                state = next_state
                steps += 1

                max_steps = config.run.max_steps_per_episode
                if terminal or (max_steps and steps >= max_steps):
                    break

            row = logger.log_episode(episode, agent.curr_step, agent.exploration_rate)
            if episode % config.run.record_every == 0:
                print(
                    "Episode {episode} | step {step} | reward {reward:.1f} | "
                    "mean100 {mean:.1f} | epsilon {epsilon:.3f}".format(
                        episode=episode,
                        step=agent.curr_step,
                        reward=row["episode_reward"],
                        mean=row["mean_reward_100"],
                        epsilon=agent.exploration_rate,
                    )
                )
            if episode % config.run.checkpoint_every == 0:
                agent.save("latest.pt")
                agent.save(f"episode-{episode:06d}.pt")

        agent.save("latest.pt")
    except KeyboardInterrupt:
        path = agent.save("latest.pt")
        print(f"Training interrupted. Saved checkpoint: {path}")
    finally:
        env.close()

    return run_dir
