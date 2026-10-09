import json
import numpy as np
import pytest
from atomworld.dataset import save_rows
from atomworld.network import fit, load_dataset, load_potential


def write_chain(path, seed, with_reference=True, kind='raw_kmc_simulation'):
    # Reversible analytic process, with all legal successors and exact boundary.
    rows=[{'x':np.array([0.]),'absorbing':False,'rates':np.array([2.]),
           'successor_x':np.array([[1.]]),'successor_absorbing':np.array([False])},
          {'x':np.array([1.]),'absorbing':False,'rates':np.array([1.,3.]),
           'successor_x':np.array([[0.],[2.]]),'successor_absorbing':np.array([False,True])},
          {'x':np.array([2.]),'absorbing':True,'rates':np.array([]),
           'successor_x':np.empty((0,1)),'successor_absorbing':np.array([],dtype=bool)}]
    if with_reference:
        for row,t in zip(rows,[1.,.5,0.]):
            row.update(reference_mean_s=t,reference_se_s=.01*t,reference_censored=0)
    save_rows(path,rows,{'data_kind':kind,'domain':{'material':'analytic','target':'state2'},'initial_seed':seed})


def test_independent_network_fit_and_checkpoint_gate(tmp_path):
    train=tmp_path/'train.npz';val=tmp_path/'val.npz';out=tmp_path/'model.pt'
    write_chain(train,1);write_chain(val,2)
    model,report=fit(train,val,out,hidden=8,iterations=200)
    assert report['residual_rmse']<1e-5
    assert report['mfpt_reference_validated']
    assert report['policy_physical_time_validated'] is False
    loaded,_=load_potential(out,{'material':'analytic','target':'state2'})
    assert float(loaded([[0.]], [False]).detach()[0])==pytest.approx(1.,rel=1e-5)
    with pytest.raises(ValueError,match='domain'):
        load_potential(out,{'material':'analytic','target':'different'})


def test_residual_fit_alone_cannot_certify_mfpt(tmp_path):
    train=tmp_path/'train.npz';val=tmp_path/'val.npz';out=tmp_path/'model.pt'
    write_chain(train,1,False);write_chain(val,2,False)
    _,report=fit(train,val,out,hidden=8,iterations=100)
    assert not report['mfpt_reference_validated']
    with pytest.raises(RuntimeError,match='has not passed'):
        load_potential(out,{'material':'analytic','target':'state2'})


def test_same_seed_and_ideal_data_rejected(tmp_path):
    train=tmp_path/'train.npz';val=tmp_path/'val.npz'
    write_chain(train,1);write_chain(val,1)
    with pytest.raises(ValueError,match='independent'):
        fit(train,val,tmp_path/'model.pt')
    write_chain(val,2,kind='ideal_schematic')
    with pytest.raises(ValueError,match='ideal'):
        load_dataset(val)
