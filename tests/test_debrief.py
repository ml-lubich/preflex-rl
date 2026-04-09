from preflex_rl.debrief import format_fallback_summary, load_env_files


def test_fallback_summary_text() -> None:
    text = format_fallback_summary(
        {
            "mean_episode_return": 10.0,
            "last_episode_return": 12.0,
            "mean_td_loss": 0.5,
            "velocity_l2_penalty": 0.0,
            "action_switch_penalty": 0.0,
        }
    )
    assert "10.00" in text or "10.0" in text
    assert "velocity" in text.lower() or "shaping" in text.lower()


def test_load_env_files_no_crash() -> None:
    load_env_files()
