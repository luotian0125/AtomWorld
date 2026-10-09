from atomworld.paths import training_root
import pytest


def test_default_training_output_is_relative_to_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('ATOMWORLD_RESULTS', raising=False)
    assert training_root('bcc')==tmp_path/'results/bcc'


def test_training_output_can_be_configured_outside_the_package(tmp_path, monkeypatch):
    monkeypatch.setenv('ATOMWORLD_RESULTS', str(tmp_path/'run/models'))
    assert training_root('fcc')==tmp_path/'run/models/fcc'
    with pytest.raises(ValueError):training_root('unknown')
