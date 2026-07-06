from pathlib import Path

import numpy as np
import pytest

from src.train.initial_state import (
    REQUIRED_ROBOT_STATE_KEYS,
    load_desk_initial_states_from_dir,
    sample_initial_states,
)


def test_two_leg_initial_state_pickles_have_expected_schema():
    state_dir = Path(__file__).resolve().parents[1] / "two_leg"
    states = load_desk_initial_states_from_dir(state_dir)

    assert len(states) == 9
    expected_mask = np.array([False, False, True, True])

    for state in states:
        assert np.asarray(state["parts_poses"]).shape == (42,)
        for key in REQUIRED_ROBOT_STATE_KEYS:
            assert key in state["robot_state"]

        metadata = state["metadata"]
        inserted_mask = np.asarray(metadata["desk_inserted_mask"], dtype=bool)
        success_mask = np.asarray(metadata["desk_success_mask"], dtype=bool)

        np.testing.assert_array_equal(inserted_mask, expected_mask)
        np.testing.assert_array_equal(success_mask, expected_mask)
        assert metadata["desk_current_pair_idx"] == 1


def test_two_leg_initial_state_sampling_is_per_env_with_replacement():
    state_dir = Path(__file__).resolve().parents[1] / "two_leg"
    states = load_desk_initial_states_from_dir(state_dir)

    sampled = sample_initial_states(states, num_envs=128, rng=np.random.default_rng(0))

    state_ids = {id(state) for state in states}
    assert len(sampled) == 128
    assert all(id(state) in state_ids for state in sampled)


def test_two_leg_yaw_initial_state_pickles_have_expected_schema_if_present():
    state_dir = Path(__file__).resolve().parents[1] / "two_leg_yaw"
    if not state_dir.exists():
        pytest.skip("two_leg_yaw initial-state directory is not present")

    states = load_desk_initial_states_from_dir(state_dir)

    assert len(states) == 10
    for state in states:
        assert np.asarray(state["parts_poses"]).shape == (42,)
        for key in REQUIRED_ROBOT_STATE_KEYS:
            assert key in state["robot_state"]

        metadata = state["metadata"]
        inserted_mask = np.asarray(metadata["desk_inserted_mask"], dtype=bool)
        success_mask = np.asarray(metadata["desk_success_mask"], dtype=bool)

        np.testing.assert_array_equal(inserted_mask, success_mask)
        assert int(success_mask.sum()) == 2
