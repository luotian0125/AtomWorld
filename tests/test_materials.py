import numpy as np
import pytest
from atomworld.adapters import make_material
from atomworld.cli import site_target, domain_metadata
from atomworld.dataset import full_event_row, enumerate_graph
from atomworld.targets import Target, TargetMemory


@pytest.mark.parametrize("material", ["bcc", "fcc", "hcp"])
def test_preview_restores_full_state_and_physical_clocks(material):
    a = make_material(material, solutes=1)
    target = site_target(a); memory = target.initialize(a)
    original = a.features().copy(); rates = a.rates().copy()
    k = a.kernel
    original_time = k.time_s if material=='hcp' else k.time
    step_count = k.step_count if material=='hcp' else k._env_step_counter
    rng = np.random.get_state()
    row = full_event_row(a, target, memory)
    np.testing.assert_array_equal(a.features(),original)
    np.testing.assert_array_equal(a.rates(),rates)
    assert (k.time_s if material=='hcp' else k.time) == original_time
    assert (k.step_count if material=='hcp' else k._env_step_counter) == step_count
    np.testing.assert_array_equal(np.random.get_state()[1],rng[1])
    assert len(row['rates']) == np.count_nonzero(rates>0)


@pytest.mark.parametrize("material", ["bcc", "fcc", "hcp"])
def test_clockless_kernel_execution_matches_energy_and_inventory(material):
    a=make_material(material, solutes=2)
    before=np.sort(a.kernel.lattice.occupancy if material=='hcp' else a.kernel.lattice.reshape(-1))
    for _ in range(6):
        event=int(np.flatnonzero(a.rates()>0)[0]); a.apply(event)
        if material=='hcp':
            assert a.kernel.time_s==0
            assert a.energy_eV()==pytest.approx(a.kernel.calculate_energy_eV())
        else:
            assert a.kernel.time==0
            assert a.energy_eV()==pytest.approx(a.kernel.calculate_system_energy(),abs=1e-8)
            np.testing.assert_allclose(a.rates(),a.kernel.calculate_diffusion_rate().reshape(-1),rtol=3e-7)
    after=np.sort(a.kernel.lattice.occupancy if material=='hcp' else a.kernel.lattice.reshape(-1))
    np.testing.assert_array_equal(before,after)


@pytest.mark.parametrize("material", ["bcc", "fcc", "hcp"])
def test_tiny_material_exact_backward_equation(material):
    a=make_material(material,solutes=1 if material=='hcp' else 0)
    graph=enumerate_graph(a,site_target(a))
    tau,_=graph.solve()
    assert tau[0]>0
    assert np.max(np.abs(graph.residuals(tau)))<1e-7


def test_truncated_graph_does_not_create_false_boundary():
    a=make_material('bcc',solutes=1); original=a.features().copy()
    with pytest.raises(RuntimeError,match='NOT a calibrated'):
        enumerate_graph(a,site_target(a),max_states=2)
    np.testing.assert_array_equal(a.features(),original)


def test_persistence_state_distinguishes_identical_microstructure():
    a=make_material('hcp',solutes=1)
    target=Target('hcp_precursor',{'sample_interval':16,'persistence_samples':4})
    assert target.key(a,TargetMemory(phase=0,streak=1)) != target.key(a,TargetMemory(phase=0,streak=2))
    assert target.key(a,TargetMemory(phase=0,streak=1)) != target.key(a,TargetMemory(phase=1,streak=1))


def test_policy_steps_do_not_borrow_kmc_time():
    for material in ('bcc','fcc'):
        a=make_material(material,solutes=1)
        event=int(np.flatnonzero(a.rates()>0)[0])
        info=a.kernel.step(event)[-1]
        assert info['kmc_delta_t']==0
        assert info['clock_owner']=='unassigned'
        assert info['physical_time_validated'] is False


def test_legacy_hcp_boolean_certificate_cannot_authorize_clock():
    from atomworld.backends.hcp.physics.clocks import PoissonCalibrationCertificate,PolicyPoissonClock
    c=PoissonCalibrationCertificate('x','y','z',True,True)
    with pytest.raises(RuntimeError,match='Legacy boolean'):
        PolicyPoissonClock(c)


def test_temperature_and_target_are_part_of_checkpoint_domain():
    a=make_material('bcc',temperature=600)
    b=make_material('bcc',temperature=650)
    target=site_target(a)
    assert domain_metadata(a,target) != domain_metadata(b,target)
