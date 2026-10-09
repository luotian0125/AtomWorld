import numpy as np
import pytest
import torch
from atomworld.residence_clock import collect, fit, read, validate, load, inputs


@pytest.fixture(scope='module')
def calibrated(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp('residence')
    train=root/'train.npz'; val=root/'val.npz'; checkpoint=root/'clock.pt'
    collect(train,size=4,cu=8,episode=200,rounds=3,seed=41,stride=1)
    collect(val,size=4,cu=8,episode=200,rounds=2,seed=61,stride=1)
    model,report=fit(train,val,checkpoint)
    return model,report,val,checkpoint


def test_passive_physical_catalogue_residence_fit_and_independent_validation(calibrated):
    model,report,val,checkpoint=calibrated
    assert report['conditional_residence_validated']
    assert report['clock_error_mean_abs_log_ratio'] < .01
    assert not report['policy_physical_time_validated']
    data,metadata=read(val)
    restored,_=load(checkpoint,metadata['domain'])
    torch.testing.assert_close(model(data['x'],data['legal']),restored(data['x'],data['legal']))


def test_clock_event_permutation_and_known_hazard_perturbation(calibrated):
    model,_,val,_=calibrated
    data,_=read(val)
    permutation=np.random.default_rng(971).permutation(32)
    torch.testing.assert_close(model(data['x'],data['legal']),
                              model(data['x'][:,permutation],data['legal'][:,permutation]))
    perturbed=dict(data);perturbed['log_total_rate']=data['log_total_rate']+np.log(2.)
    result=validate(model,perturbed)
    assert result['clock_error_mean_abs_log_ratio'] > .68
    assert not result['conditional_residence_validated']


def test_known_rates_and_actor_inputs_do_not_enter_clock_features():
    features=np.zeros((4,8,9));features[...,0]=1;features[...,1]=1;features[...,3]=.1
    first,mask=inputs(features)
    features[...,5:]=np.random.default_rng(81).normal(size=(4,8,4))
    second,other_mask=inputs(features)
    assert np.array_equal(first,second)
    assert np.array_equal(mask,other_mask)


def test_clock_checkpoint_rejects_temperature_domain_change(calibrated):
    _,_,val,checkpoint=calibrated
    _,metadata=read(val);wrong=dict(metadata['domain']);wrong['temperature_K']=700
    with pytest.raises(ValueError,match='domain'):load(checkpoint,wrong)


def test_clock_checkpoint_rejects_failed_validation(calibrated, tmp_path):
    _,_,val,checkpoint=calibrated
    _,metadata=read(val)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    payload['validation']['conditional_residence_validated']=False
    failed=tmp_path/'failed.pt';torch.save(payload,failed)
    with pytest.raises(ValueError,match='validation'):load(failed,metadata['domain'])
