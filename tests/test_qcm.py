from dataclasses import replace
import numpy as np
import pytest
from safecad import Config, Features, SafeCAD
from safecad.core import Observation, Prototype
from safecad.features import coreset_indices, context, nearest, normalize, retained_indices, top_mean


def feature(angle=0.0, cg=1.0, cp=1.0, coords=None):
    z = np.array([[np.cos(angle), np.sin(angle)]], dtype=np.float32)
    return Features(z, [[0.5, 0.5]] if coords is None else coords, cg, [cp])


def detector(**kwargs):
    model = SafeCAD(Config(**kwargs))
    model.commission("camera", [feature()], [feature() for _ in range(20)])
    return model


def test_default_lifecycle_and_preupdate_prediction():
    model = detector()
    state = model.domains["camera"]
    original_anchor = state.anchor.copy()
    shifted = feature(0.12)
    for episode in range(8):
        result = model.observe(shifted, episode, "camera")
        assert result.score > 0
    assert len(state.pending) == 1
    assert len(state.active) == 0
    assert state.pending[0].created == 1
    assert state.pending[0].episode_count == 0
    for episode in range(8, 12):
        model.observe(shifted, episode, "camera")
    assert len(state.active) == 0  # Enough episodes, but only one future window.
    assert state.pending[0].episode_count == 3
    assert state.pending[0].window_count == 1
    for episode in range(12, 16):
        result = model.observe(shifted, episode, "camera")
    assert result.score > 0  # The boundary observation is scored before activation.
    assert len(state.active) == 1
    assert model.predict(shifted, "camera").score == 0
    assert model.predict(shifted, "camera").anomalous is False
    assert state.threshold == 0
    np.testing.assert_array_equal(state.anchor, original_anchor)
    assert state.anchor.flags.writeable is False
    assert len(state.active[0].episodes) == len(state.active[0].windows) == 0


def test_repeated_frames_cannot_vote_as_distinct_episodes():
    model = detector(episode_cap=5)
    for episode in range(2):
        for frame in range(5):
            model.observe(feature(0.12), episode, "camera", end_episode=frame == 4)
    assert not model.domains["camera"].pending
    assert model.domains["camera"].completed_window == -1
    for episode in range(2, 4):
        model.observe(feature(0.12), episode, "camera")
    assert not model.domains["camera"].pending  # Still only one completed window.


@pytest.mark.parametrize("cg,cp", [(0.5, 1), (1, 0.5)])
def test_inconsistent_patches_never_enter_pending(cg, cp):
    model = detector()
    for episode in range(20):
        model.observe(feature(0.12, cg, cp), episode, "camera")
    assert not model.domains["camera"].pending
    assert not model.domains["camera"].active


def test_isolated_local_defect_not_admitted():
    model = detector()
    normal = Features(normalize([[1, 0], [1, 0], [1, 0], [1, 0]]),
                      [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]], 1, np.ones(4))
    defective = Features(normalize([[1, 0], [1, 0], [1, 0], [0.8, 0.6]]), normal.coords, 1, np.ones(4))
    for episode in range(20):
        model.observe(defective if episode == 0 else normal, episode, "camera")
    assert all(not (p.arrival == 1 and p.patch_index == 3)
               for p in model.domains["camera"].pending + model.domains["camera"].active)


def test_recurrent_defect_can_activate_documented_limitation():
    model = detector(route_radius=0.01)
    for episode in range(16):
        model.observe(feature(0.6), episode, "camera")
    assert model.predict(feature(0.6), "camera").score == 0


def test_spatial_support_and_exchangeable_mode():
    registered = detector()
    bulk = detector(spatial_radius=np.sqrt(2))
    for episode in range(12):
        x = feature(0.12, coords=[[0.1 if episode == 0 else 0.9, 0.5]])
        registered.observe(x, episode, "camera")
        bulk.observe(x, episode, "camera")
    assert not any(p.arrival == 1 for p in registered.domains["camera"].pending)
    assert any(p.arrival == 1 for p in bulk.domains["camera"].pending)


def test_routing_abstains_and_retention_is_bounded():
    model = detector(unsupported_capacity=3, unsupported_lifetime=1)
    threshold = model.domains["camera"].threshold
    for episode in range(20):
        result = model.observe(feature(1.0), episode)
        assert result.abstained and not result.calibrated
        assert result.threshold is None and result.anomalous is None
        assert result.nearest_domain == "camera"
        assert len(model.unsupported) <= 3
    assert not model.domains["camera"].quarantine
    assert model.domains["camera"].threshold == threshold


def test_episode_routing_fixed_by_first_frame():
    model = detector(episode_cap=2)
    first = model.observe(feature(0.1), 0, end_episode=False)
    second = model.observe(feature(1), 0)
    assert first.domain == second.domain == "camera"
    assert len(model.domains["camera"].quarantine) == 2
    with pytest.raises(ValueError):
        model.observe(feature(), 0)


def test_quarantine_expiration_before_evidence():
    model = detector(quarantine_lifetime=1)
    for episode in range(20):
        model.observe(feature(0.12), episode, "camera")
    assert len(model.domains["camera"].quarantine) == 4
    assert not model.domains["camera"].pending


def test_pending_can_activate_on_final_lifetime_boundary():
    model = detector(pending_lifetime=2)
    for episode in range(16):
        model.observe(feature(0.12), episode, "camera")
    assert len(model.domains["camera"].active) == 1


def test_pending_expires_without_future_eligible_matches():
    model = detector(pending_lifetime=2)
    for episode in range(8):
        model.observe(feature(0.12), episode, "camera")
    for episode in range(8, 16):
        model.observe(feature(0.12, cp=0.5), episode, "camera")
    assert not model.domains["camera"].active
    assert all(p.created > 1 for p in model.domains["camera"].pending)


def test_old_window_reevaluation_does_not_credit_activation():
    model = detector()
    for episode in range(8):
        model.observe(feature(0.12), episode, "camera")
    for episode in range(8, 12):
        model.observe(feature(0.12, cp=0.5), episode, "camera")
    p = model.domains["camera"].pending[0]
    assert p.episode_count == p.window_count == 0


def test_threshold_uses_paper_order_statistic_and_strict_comparison():
    model = SafeCAD()
    init = [feature()]
    calibration = [feature(v) for v in np.linspace(0.01, 0.2, 20)]
    model.commission("camera", init, calibration)
    expected = np.linalg.norm(calibration[18].z[0]-init[0].z[0])
    assert model.domains["camera"].threshold == pytest.approx(expected)
    assert model.predict(calibration[18], "camera").anomalous is False
    assert model.predict(calibration[19], "camera").anomalous is True


def test_activation_cannot_raise_scores_above_anchor():
    model = detector()
    for episode in range(16):
        model.observe(feature(0.12), episode, "camera")
    for angle in np.linspace(-1, 1, 100):
        p = model.predict(feature(angle), "camera")
        assert np.all(p.patch_scores <= p.anchor_scores)


def test_context_update_deduplicates_origin_observations():
    model = detector()
    for episode in range(16):
        model.observe(feature(0.12), episode, "camera")
    expected = normalize(0.95*np.array([1, 0])+0.05*feature(0.12).z[0])
    np.testing.assert_allclose(model.domains["camera"].adaptive_context, expected, atol=1e-7)


def test_checkpoint_resume_matches_uninterrupted_mid_episode(tmp_path):
    model = detector(episode_cap=2)
    for episode in range(8):
        model.observe(feature(0.12), episode, "camera")
    model.observe(feature(0.12), 8, "camera", end_episode=False)
    path = tmp_path/"model.safecad"
    model.save(path)
    restored = SafeCAD.load(path)
    model.observe(feature(0.12), 8, "camera")
    restored.observe(feature(0.12), 8, "camera")
    for episode in range(9, 20):
        a = model.observe(feature(0.12), episode, "camera")
        b = restored.observe(feature(0.12), episode, "camera")
        assert a.score == b.score
    assert model.last_boundary == restored.last_boundary
    assert restored.domains["camera"].anchor.flags.writeable is False
    assert restored.domains["camera"].active[0].key == model.domains["camera"].active[0].key


def test_caps_and_rollback():
    model = detector(pending_capacity=2, active_capacity=2, quarantine_capacity=8, duplicate_radius=0.001)
    for episode in range(48):
        model.observe(feature(0.08+episode % 3*0.015), episode, "camera")
        s = model.domains["camera"]
        assert len(s.pending) <= 2 and len(s.active) <= 2 and len(s.quarantine) <= 8
        for p in s.pending:
            assert len(p.episodes) <= 3 and len(p.windows) <= 2
    model.rollback("camera")
    assert not s.active and not s.pending and not s.quarantine
    np.testing.assert_array_equal(s.initial_context, s.adaptive_context)


def test_no_support_from_pending_itself():
    model = detector()
    state = model.domains["camera"]
    p = Prototype(1, 0, 0, 0, 0, feature(0.12).z[0], np.array([0.5, 0.5]), feature(0.12).z[0])
    state.pending = [p]
    for episode in range(4):
        model.observe(feature(0.12), episode, "camera")
    assert p.episode_count == 0 and not state.active


def test_coreset_oldest_and_origin_ties():
    z = np.array([[0, 0], [1, 0], [-1, 0], [0, 1]], np.float32)
    assert coreset_indices(z, 2, [(0, 0), (3, 0), (1, 0), (2, 0)]) == [0, 2]
    assert coreset_indices(z, 4) == [0, 1, 2, 3]
    assert coreset_indices(np.zeros((3, 2)), 3) == [0, 1, 2]


def test_retention_cell_centers_and_row_major_ties():
    coords = np.array([[0.2, 0.2], [0.3, 0.2], [0.7, 0.7]], np.float32)
    np.testing.assert_array_equal(retained_indices(coords, 2, 2), [0, 2])
    assert top_mean(np.array([0, 1, 2, 3]), 0.26) == 2.5
    np.testing.assert_array_equal(nearest(np.ones((2, 2)), np.ones((3, 2))), [0, 0])


def test_channelwise_trimmed_context():
    z = np.array([[1, 20], [2, 3], [3, 2], [20, 1]], np.float32)
    np.testing.assert_allclose(context(z, 0.25), normalize([2.5, 2.5]))


def test_commissioning_and_consistency_validation():
    model = SafeCAD()
    with pytest.raises(ValueError):
        model.commission("camera", [feature()], [feature() for _ in range(19)])
    model = detector(max_domains=1)
    with pytest.raises(ValueError):
        model.commission("other", [feature()], [feature() for _ in range(20)])
    with pytest.raises(ValueError):
        model.observe(Features([[1, 0]], [[0.5, 0.5]]), 0, "camera")
    with pytest.raises(ValueError):
        Config(activation_windows=9)


def test_prediction_is_read_only():
    model = detector()
    for _ in range(10):
        model.predict(feature(0.12), "camera")
    assert model.arrival == 0
    assert not model.domains["camera"].quarantine


def test_unretained_frames_need_only_single_view_features():
    model = detector()
    assert model.needs_consistency()
    model.observe(feature(0.12), 0, "camera", end_episode=False)
    assert not model.needs_consistency()
    x = feature(0.12)
    prediction = model.observe(Features(x.z, x.coords), 0, "camera")
    assert prediction.calibrated
    assert len(model.domains["camera"].quarantine) == 1
    assert model.needs_consistency()
