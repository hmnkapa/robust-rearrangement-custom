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


def test_random_small_triggers_on_internal_interval():
    runner = PerturbRunner("random_small")
    runner.perturb_per_timesteps = 2
    runner.perturb_max_force = 2.0
    runner.reset_episode(num_envs=3, device=torch.device("cpu"))

    first = runner.compute_force(_context(0, [None, None, None]))
    second = runner.compute_force(_context(1, [None, None, None]))
    third = runner.compute_force(_context(2, [None, None, None]))

    assert torch.linalg.norm(first, dim=-1).gt(0).all()
    assert torch.linalg.norm(second, dim=-1).eq(0).all()
    assert torch.linalg.norm(third, dim=-1).gt(0).all()
    assert torch.linalg.norm(first, dim=-1).le(2.0).all()


def test_short_large_waits_delay_and_fires_once():
    runner = PerturbRunner("short_large")
    runner.perturb_state = "place"
    runner.perturb_delay = 1
    runner.perturb_min_force = 4.0
    runner.perturb_max_force = 4.0
    runner.reset_episode(num_envs=2, device=torch.device("cpu"))

    first = runner.compute_force(_context(0, ["base-leg-place", "pick"]))
    second = runner.compute_force(_context(1, ["base-leg-place", "pick"]))
    third = runner.compute_force(_context(2, ["base-leg-place", "pick"]))

    assert torch.linalg.norm(first, dim=-1).eq(0).all()
    assert torch.isclose(torch.linalg.norm(second[0]), torch.tensor(4.0))
    assert torch.linalg.norm(second[1]).eq(0)
    assert torch.linalg.norm(third, dim=-1).eq(0).all()


def test_place_slowdown_opposes_downward_velocity():
    runner = PerturbRunner("place_slowdown")
    runner.perturb_max_force = 3.0
    runner.perturb_slowdown_gain = 10.0
    runner.perturb_down_vel_threshold = 0.0
    runner.reset_episode(num_envs=3, device=torch.device("cpu"))
    ee_pos_vel = torch.tensor(
        [
            [0.0, 0.0, -0.2],
            [1.0, 0.0, -0.5],
            [0.0, 0.0, 0.1],
        ]
    )

    forces = runner.compute_force(
        _context(0, ["base-leg-place", "place", "base-leg-place"], ee_pos_vel)
    )

    assert torch.dot(forces[0], ee_pos_vel[0]) < 0
    assert torch.dot(forces[1], ee_pos_vel[1]) < 0
    assert torch.linalg.norm(forces[2]).eq(0)
    assert torch.linalg.norm(forces, dim=-1).le(3.0).all()
