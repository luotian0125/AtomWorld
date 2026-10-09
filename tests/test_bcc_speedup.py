import pytest
from atomworld.bcc_af import first_passage
from atomworld.bcc_speedup import paired_speedup, aggregate


def test_speedup_uses_reference_first_hit_not_reference_total_budget():
    reference=first_passage([10.,8.,8.,8.,8.],8.,4)
    method=first_passage([10.,9.,8.,8.,8.],8.,4)
    pair=paired_speedup(reference,method,2.,8.)
    assert method['af']==2.
    assert pair['first_passage_step_speedup']==.5
    assert pair['fixed_budget_evaluation_wall_speedup']==.25


def test_missing_hits_keep_upper_bound_and_contribute_to_capped_cost():
    reference=first_passage([10.,8.,8.],8.,2)
    hit=paired_speedup(reference,first_passage([10.,8.,8.],8.,2),1.,2.)
    miss=paired_speedup(reference,first_passage([10.,9.,9.],8.,2),1.,2.)
    assert miss['first_passage_step_speedup'] is None
    assert miss['first_passage_step_speedup_upper_bound']==.5
    summary=aggregate([hit,miss])
    assert summary['first_passage_step_speedup_mean_reached']==1.
    assert summary['restricted_mean_first_passage_step_ratio']==pytest.approx(2/3)
    assert summary['method_reached_count']==1 and summary['total_pair_count']==2


def test_wall_ratio_is_ratio_of_ensemble_costs_not_mean_of_ratios():
    reference=first_passage([2.,1.],1.,1)
    pairs=[paired_speedup(reference,reference,1.,1.),paired_speedup(reference,reference,1.,9.)]
    assert aggregate(pairs)['fixed_budget_evaluation_wall_speedup']==.2


def test_mismatched_target_and_invalid_wall_cost_are_rejected():
    reference=first_passage([2.,1.],1.,1)
    with pytest.raises(ValueError): paired_speedup(reference,first_passage([2.,0.],0.,1),1.,1.)
    with pytest.raises(ValueError): paired_speedup(reference,reference,1.,0.)
