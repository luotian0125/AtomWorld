import numpy as np
import pytest
import torch
from atomworld.poisson import FiniteRateGraph, signed_progress, normalized_residual
from atomworld.clocks import PassiveKMCClock, PotentialProgress
from atomworld.network import PoissonMFPTNetwork, backward_residual


def reversible_graph():
    return FiniteRateGraph([np.array([2.]), np.array([1., 3.]), np.array([])],
                           [np.array([1]), np.array([0, 2]), np.array([], dtype=int)],
                           np.array([False, False, True]))


def test_exact_reversible_mfpt_and_moments():
    g = reversible_graph()
    tau, second = g.solve()
    np.testing.assert_allclose(tau, [1., .5, 0.], rtol=1e-14)
    np.testing.assert_allclose(g.residuals(tau), 0, atol=1e-14)
    np.testing.assert_allclose(second, [5/3, 2/3, 0], atol=1e-14)


def test_paper_increment_is_signed_and_telescopes_on_cycle():
    tau, _ = reversible_graph().solve()
    diagnostic = PotentialProgress()
    assert diagnostic.advance(tau[0], tau[1]) == pytest.approx(.5)
    assert diagnostic.advance(tau[1], tau[0]) == pytest.approx(-.5)
    assert diagnostic.progress_s == pytest.approx(0.)
    assert not hasattr(diagnostic, "elapsed_s")
    # Ordinary residence times for these two events have positive mean .75 s.
    assert 1/2 + 1/4 > diagnostic.progress_s


def test_zero_rate_absorbing_boundary_needs_no_division():
    g = FiniteRateGraph([np.array([5.]), np.array([])], [np.array([1]), np.array([], dtype=int)], [False, True])
    tau, _ = g.solve()
    assert signed_progress(tau[0], tau[1]) == pytest.approx(.2)
    assert normalized_residual(tau[0], tau[[1]], np.array([5.])) == pytest.approx(0)


def test_unreachable_closed_class_is_rejected():
    with pytest.raises(ValueError, match="unreachable"):
        FiniteRateGraph([[1.], [1.], []], [[1], [0], []], [False, False, True])


@pytest.mark.parametrize("bad", [[-1.], [float('nan')], [float('inf')]])
def test_invalid_rates_rejected(bad):
    with pytest.raises(ValueError):
        FiniteRateGraph([bad, []], [[1], []], [False, True])


def test_nonabsorbing_dead_state_rejected():
    with pytest.raises(ValueError, match="dead state"):
        FiniteRateGraph([[], []], [[], []], [False, True])


def test_mc_mean_matches_exact_uncertainty():
    g = reversible_graph(); tau, second = g.solve()
    rng = np.random.default_rng(91)
    samples = np.array([g.sample_hitting_time(0, rng) for _ in range(20000)])
    expected_se = np.sqrt((second[0]-tau[0]**2)/len(samples))
    assert abs(samples.mean()-tau[0]) < 5*expected_se


def test_passive_clock_owns_event_and_exponential_draws():
    clock = PassiveKMCClock(31)
    samples = np.array([clock.draw([2., 8.]) for _ in range(30000)])
    assert abs(samples[:,0].mean()-.8) < .012
    assert abs(samples[:,1].mean()-.1) < .003
    assert samples[:,1].min() >= 0
    assert clock.elapsed_s == pytest.approx(samples[:,1].sum())


def test_network_boundary_is_exact_zero_and_operator_uses_all_rates():
    model = PoissonMFPTNetwork(1, 4)
    # A constant nonabsorbing potential of .2 solves one rate-5 jump to target.
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    model.layers[-1].bias.data.fill_(np.log(np.expm1(.2)))
    data = {"x": torch.tensor([[0.], [1.]],dtype=torch.float64),
            "absorbing": torch.tensor([False,True]),
            "successor_x": torch.tensor([[[1.]],[[0.]]],dtype=torch.float64),
            "successor_absorbing": torch.tensor([[True],[True]]),
            "rates": torch.tensor([[5.],[0.]],dtype=torch.float64)}
    assert model(data["x"],data["absorbing"])[1].item() == 0
    assert backward_residual(model,data).abs().max().item() < 1e-14
    assert model.dimensionless_u(data["x"],data["absorbing"], [5.,0.])[1].item() == 0
