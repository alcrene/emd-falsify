from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
import pytest

from emdd.utils import ParamColl, expand

def test_ParamColl():

    @dataclass
    class DataParamset(ParamColl):
        L: int
        λ: float
        σ: float
        δy: float

    @dataclass
    class ModelParamset(ParamColl):
        λ: float
        σ: float
        #μ: float

    #
    data_params = DataParamset(
        L=400,
        λ=1,
        σ=1,
        δy=expand([-1, -0.3, 0, 0.3, 1])
    )
    model_params = ModelParamset(
        λ=expand(np.logspace(-1, 1, 10)),
        σ=expand(np.linspace(0.1, 3, 8))
    )
    model_params_aligned = ModelParamset(
        λ=expand(np.logspace(-1, 1, 10)),
        σ=expand(np.linspace(0.1, 3, 10))
    )

    #
    # Iterating over ParamColl returns the keys
    assert len(list(data_params)) == len(data_params.keys()) == 4
    assert list(data_params) == ["L", "λ", "σ", "δy"]

    # Expanding a list
    assert list(data_params.inner()) == list(data_params.outer())
    assert len(list(data_params.outer())) == len(data_params.δy) == data_params.outer_len == 5

    # Expanding an array + Non-aligned doesn't allow inner() iterator
    assert len(list(model_params.outer())) == 10*8
    with pytest.raises(ValueError):
        next(model_params.inner())

    # Expanding an array + Aligned expanded params allows inner() iterator
    assert len(list(model_params_aligned.inner())) == len(model_params_aligned.λ) == model_params_aligned.inner_len == 10
    assert len(list(model_params_aligned.outer())) == model_params_aligned.outer_len == 10*10

    assert dict(**data_params) == {k: v for k,v in asdict(data_params).items()
                                      if not k.startswith("_") and k not in {"seed"}}

    # Slicing inner() and outer() works as advertised
    assert len(list(model_params_aligned.inner(2, 8))) == 6
    assert len(list(model_params_aligned.inner(2, 8, 2))) == 3
    assert len(list(model_params_aligned.outer(5,20)))   == 15
    assert len(list(model_params_aligned.outer(5,20,5))) == 3


    ## Param collections with random generators ##

    # Random params require a seed

    with pytest.raises(TypeError):
        DataParamset(
            L=400,
            λ=stats.norm(),
            σ=1,
            δy=expand([-1, -0.3, 0, 0.3, 1])
        )


    data_params = DataParamset(
        L=400,
        λ=stats.norm(),
        σ=expand([1, 0.2, 0.05]),
        δy=expand([-1, -0.3, 0, 0.3, 1]),
        seed=314
    )

    # σ and δy are not aligned: cannot do inner product
    with pytest.raises(ValueError):
        list(data_params.inner())  # `list()` is required to trigger error
    assert len(list(data_params.outer())) == data_params.outer_len == 3*5

    # Now align σ and δy: can do both inner and outer product
    data_params.σ = expand([10, 0, 1, 0.2, 0.05])
    assert len(list(data_params.inner())) == data_params.inner_len == 5
    assert len(list(data_params.outer())) == data_params.outer_len == 5*5

    # Randomly generated values are reproducible
    ival1 = tuple(p.λ for p in data_params.inner())
    ival2 = tuple(p.λ for p in data_params.inner())
    assert ival1 == ival2

    oval1 = tuple(p.λ for p in data_params.outer())
    oval2 = tuple(p.λ for p in data_params.outer())
    assert oval1 == oval2

    # Values change if seed changes
    data_params.seed = 628
    ival3 = tuple(p.λ for p in data_params.inner())
    oval3 = tuple(p.λ for p in data_params.outer())
    assert ival1 != ival3
    assert oval1 != oval3

    # When there are only random and scalar values, only `inner` is possible
    data_params.σ = stats.lognorm(1)
    data_params.δy = stats.uniform(-1, 1)
    with pytest.raises(ValueError):
        list(zip(data_params.outer(), range(10)))
    it = data_params.inner()  # Infinite iterator
    old_p = next(it)
    for _ in range(100):
        p = next(it)
        assert p.σ != old_p.σ  # All generated values are different
        old_p = p
