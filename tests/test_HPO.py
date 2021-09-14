import pytest
from HPO.utils import weight_freezing
import torch.nn as nn
from HPO.utils.model_constructor import Model
from HPO.searchspaces.OOP_config import init_config as _config

def generate_model( input_size : int, output_size : int):


    config = _config()
    hyperparameters = config.sample_configuration()
    model = Model( input_size, output_size, hyperparameters )

def test_normal_weight_freezing(model : Model):
    #get input size

    #generate random input data
    #check if weights in normal cells have changed

    frozen_model = weight_freezing.freeze_normal_cells( model ) 


def test_empty_slap():
    assert slap_many(LikeState.empty, '') is LikeState.empty


def test_single_slaps():
    assert slap_many(LikeState.empty, 'l') is LikeState.liked
    assert slap_many(LikeState.empty, 'd') is LikeState.disliked


@pytest.mark.parametrize("test_input,expected", [
    ('ll', LikeState.empty),
    ('dd', LikeState.empty),
    ('ld', LikeState.disliked),
    ('dl', LikeState.liked),
    ('ldd', LikeState.empty),
    ('lldd', LikeState.empty),
    ('ddl', LikeState.liked),
])
def test_multi_slaps(test_input, expected):
    assert slap_many(LikeState.empty, test_input) is expected


@pytest.mark.skip(reason="regexes not supported yet")
def test_regex_slaps():
    assert slap_many(LikeState.empty, '[ld]*ddl') is LikeState.liked


@pytest.mark.xfail
def test_divide_by_zero():
    assert 1 / 0 == 1


def test_invalid_slap():
    with pytest.raises(ValueError):
        slap_many(LikeState.empty, 'x')


@pytest.mark.xfail
def test_db_slap(db_conn):
    db_conn.read_slaps()
    assert ...


def test_print(capture_stdout):
    print("hello")
    assert capture_stdout["stdout"] == "hello\n"