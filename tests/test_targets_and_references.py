import numpy as np
import pytest
from atomworld.adapters import make_material
from atomworld.cli import site_target
from atomworld.dataset import passive_reference
from atomworld.targets import Target, TargetMemory, paper_target
from atomworld.runtime import PassiveKMCSession


def test_censored_reference_is_not_a_success_only_mfpt():
    a=make_material('bcc',solutes=0)
    target=site_target(a);memory=target.initialize(a)
    result=passive_reference(a,target,memory,20,31,1)
    assert result['reference_censored']>0
    assert np.isnan(result['reference_mean_s'])
    assert np.isnan(result['reference_se_s'])


def test_exact_hcp_persistence_sampling_phase_and_thresholds():
    from atomworld.backends.hcp.physics.local_energy import LocalEnergyModel,PairInteractionShell
    from atomworld.backends.hcp.analysis.clusters import hydrogen_cluster_metrics
    a=make_material('hcp',solutes=2)
    # Synthetic all-distance attraction only tests boundary mechanics; no
    # synthetic values are promoted to material data or a calibration artifact.
    shell=PairInteractionShell(0.,100.,-.01,'T','T','synthetic_test')
    a.kernel.energy_model=LocalEnergyModel(pair_shells=[shell])
    target=Target('hcp_precursor',{'sample_interval':16,'persistence_samples':4})
    memory=target.initialize(a)
    assert memory.streak==1 and not memory.hit
    for _ in range(47):memory=target.advance(a,memory)
    assert memory.streak==3 and not memory.hit
    memory=target.advance(a,memory)
    assert memory.hit


def test_bcc_vacancy_arrival_target_uses_periodic_connected_cluster():
    a=make_material('bcc',size=(3,3,3),solutes=4)
    k=a.kernel
    # Build a traced, synthetic topology fixture (not a material result).
    k.cu_pos=np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
    k.vacancy_pos=np.array([[5,5,5]])
    assert Target('bcc_arrival',{'min_cluster_size':4}).initialize(a).hit
    assert not Target('bcc_arrival',{'min_cluster_size':5}).initialize(a).hit


def test_fcc_persistence_mask_is_markov_state():
    a=make_material('fcc',solutes=0)
    target=Target('fcc_nucleus',{'sample_interval':2})
    m1=TargetMemory(previous_mask=np.zeros(a.kernel.size,dtype=bool))
    m2=TargetMemory(previous_mask=np.ones(a.kernel.size,dtype=bool))
    assert target.key(a,m1)!=target.key(a,m2)


def test_impossible_default_targets_fail_early():
    with pytest.raises(ValueError,match='at least 4'):
        paper_target('bcc',make_material('bcc',solutes=1))
    with pytest.raises(ValueError,match='sourced attractive'):
        paper_target('hcp',make_material('hcp',solutes=1))


def test_passive_session_time_is_positive_without_changing_kernel_clock():
    a=make_material('bcc',solutes=0)
    session=PassiveKMCSession(a,site_target(a),31)
    row=session.step()
    assert row['physical_delta_time_s']>=0
    assert row['clock_owner']=='passive_kmc'
    assert a.kernel.time==0
