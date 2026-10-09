import numpy as np
import pytest
from atomworld.bcc_af import first_passage, aggregate


def test_first_crossing_uses_event_index_and_keeps_transient_hit():
    record = first_passage([10., 9., 8., 9., 7.], 8., 20)
    assert record['first_hit_step'] == 2
    assert record['af'] == 10.


def test_equality_and_final_step_are_hits():
    record = first_passage([3., 2., 1.], 1., 2)
    assert record['first_hit_step'] == 2 and record['af'] == 1.


def test_unreached_is_censored_and_not_imputed_in_mean():
    missed = first_passage([10., 9., 8.], 7., 2)
    assert missed['af'] is None and missed['af_upper_bound'] == 1.
    summary = aggregate([first_passage([10., 7., 6.], 7., 2), missed])
    assert summary['af_mean_reached'] == 2.
    assert summary['af_censored_count'] == 1 and summary['af_reached_count'] == 1
    assert summary['af_se_reached'] is None


def test_initial_satisfaction_is_not_infinite_average():
    record = first_passage([5., 4.], 6., 1)
    assert record['first_hit_step'] == 0 and record['af'] is None
    summary = aggregate([record])
    assert summary['af_initial_satisfied_count'] == 1 and summary['af_mean_reached'] is None


def test_reference_self_first_hit_can_precede_budget():
    energy = [10., 7., 6., 8., 7.]
    assert first_passage(energy, energy[-1], len(energy)-1)['af'] == 4.


def test_aggregate_reports_se_over_observed_finite_values():
    values = [first_passage([10., 8., 6.], 8., 4), first_passage([10., 8., 6.], 6., 4)]
    summary = aggregate(values)
    assert summary['af_mean_reached'] == 3. and summary['af_se_reached'] == 1.


@pytest.mark.parametrize('energy,target,budget', [([1.,np.nan],0.,2), ([1.],0.,2), ([1.,0.],np.inf,2), ([1.,0.],0.,0)])
def test_invalid_input_is_rejected(energy,target,budget):
    with pytest.raises(ValueError): first_passage(energy,target,budget)
