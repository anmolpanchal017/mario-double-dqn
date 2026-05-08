from mario_dqn.logging import MetricLogger


def test_metric_logger_writes_csv(tmp_path):
    logger = MetricLogger(tmp_path)
    logger.log_step(1.0, 0.5, 2.0)
    row = logger.log_episode(episode=0, step=1, epsilon=0.9)

    assert row["episode_reward"] == 1.0
    metrics = tmp_path / "metrics.csv"
    assert metrics.exists()
    assert "episode_reward" in metrics.read_text(encoding="utf-8")

