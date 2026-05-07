import torch

from src.eval.perturb_util import PerturbContext, PerturbRunner


def _context(step_idx, skill_states, ee_pos_vel=None):
    return PerturbContext(
        step_idx=step_idx,
        num_envs=len(skill_states),
        device=torch.device("cpu"),
        furniture_name="round_table",
        task_name="round_table",
        skill_states=skill_states,
        ee_pos_vel=ee_pos_vel,
    )


# -- random_small -----------------------------------------------------------


def test_random_small_triggers_on_internal_interval():
    runner = PerturbRunner("random_small")
    runner.random_small_interval = 2
    runner.random_small_max_force = 2.0
    runner.reset_episode(num_envs=3, device=torch.device("cpu"))

    first = runner.compute_force(_context(0, [None, None, None]))
    second = runner.compute_force(_context(1, [None, None, None]))
    third = runner.compute_force(_context(2, [None, None, None]))

    assert torch.linalg.norm(first, dim=-1).gt(0).all()
    assert torch.linalg.norm(second, dim=-1).eq(0).all()
    assert torch.linalg.norm(third, dim=-1).gt(0).all()
    assert torch.linalg.norm(first, dim=-1).le(2.0).all()


def test_random_small_applies_force():
    runner = PerturbRunner("random_small")
    assert runner.applies_force
    assert not runner.modifies_action


# -- short_large ------------------------------------------------------------


def test_short_large_waits_delay_and_fires_once():
    runner = PerturbRunner("short_large")
    runner.short_large_trigger_state = "place"
    runner.short_large_delay = 1
    runner.short_large_min_force = 4.0
    runner.short_large_max_force = 4.0
    runner.reset_episode(num_envs=2, device=torch.device("cpu"))

    first = runner.compute_force(_context(0, ["base-leg-place", "pick"]))
    second = runner.compute_force(_context(1, ["base-leg-place", "pick"]))
    third = runner.compute_force(_context(2, ["base-leg-place", "pick"]))

    assert torch.linalg.norm(first, dim=-1).eq(0).all()
    assert torch.isclose(torch.linalg.norm(second[0]), torch.tensor(4.0))
    assert torch.linalg.norm(second[1]).eq(0)
    assert torch.linalg.norm(third, dim=-1).eq(0).all()


def test_short_large_applies_force():
    runner = PerturbRunner("short_large")
    assert runner.applies_force
    assert not runner.modifies_action


# -- place_slowdown (action modification) -----------------------------------


def test_place_slowdown_scales_z_during_place():
    runner = PerturbRunner("place_slowdown")
    runner.place_slowdown_speed_ratio = 0.3
    runner.place_slowdown_random_force = 0.0
    runner.reset_episode(num_envs=3, device=torch.device("cpu"))

    action = torch.tensor([
        [0.1, 0.2, -0.5],
        [0.3, 0.1, -0.4],
        [0.0, 0.0, -0.3],
    ])

    # env 0, 2 in place; env 1 in pick
    modified = runner.modify_action(
        action, _context(0, ["base-leg-place", "pick", "place"])
    )

    # env 0: z scaled by 0.3
    assert torch.isclose(modified[0, 0], torch.tensor(0.1))
    assert torch.isclose(modified[0, 1], torch.tensor(0.2))
    assert torch.isclose(modified[0, 2], torch.tensor(-0.15))
    # env 1: not in place, unchanged
    assert torch.equal(modified[1], action[1])
    # env 2: z scaled by 0.3
    assert torch.isclose(modified[2, 2], torch.tensor(-0.09))


def test_place_slowdown_no_random_force_by_default():
    runner = PerturbRunner("place_slowdown")
    assert not runner.applies_force  # random_force defaults to 0.0
    assert runner.modifies_action


def test_place_slowdown_with_random_force():
    runner = PerturbRunner("place_slowdown")
    runner.place_slowdown_random_force = 2.0
    runner.place_slowdown_random_interval = 2
    runner.reset_episode(num_envs=3, device=torch.device("cpu"))

    assert runner.applies_force
    assert runner.modifies_action

    first = runner.compute_force(_context(0, ["place", "place", "place"]))
    second = runner.compute_force(_context(1, ["place", "place", "place"]))
    third = runner.compute_force(_context(2, ["place", "place", "place"]))

    assert torch.linalg.norm(first, dim=-1).gt(0).all()
    assert torch.linalg.norm(second, dim=-1).eq(0).all()
    assert torch.linalg.norm(third, dim=-1).gt(0).all()
    assert torch.linalg.norm(first, dim=-1).le(2.0).all()


def test_place_slowdown_speed_ratio_one_is_noop():
    runner = PerturbRunner("place_slowdown")
    runner.place_slowdown_speed_ratio = 1.0
    runner.reset_episode(num_envs=2, device=torch.device("cpu"))

    action = torch.tensor([[0.1, 0.2, -0.5], [0.3, 0.1, -0.4]])
    modified = runner.modify_action(action, _context(0, ["place", "place"]))
    assert torch.equal(modified, action)


# -- stats ------------------------------------------------------------------


def test_stats_tracks_action_modifications():
    runner = PerturbRunner("place_slowdown")
    runner.place_slowdown_speed_ratio = 0.5
    runner.reset_episode(num_envs=4, device=torch.device("cpu"))

    action = torch.zeros(4, 3)
    runner.modify_action(action, _context(0, ["place", "pick", "place", "place"]))

    s = runner.stats.summary()
    assert s["modified_steps"] == 1
    assert s["modified_env_steps"] == 3
