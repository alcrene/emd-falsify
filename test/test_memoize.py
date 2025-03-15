from functools import cache
from emdcmp.memoize import nofail_functools_cache

def test_nofail_cache(caplog):
    @nofail_functools_cache
    @cache
    def my_slow_function(x):
        return x*2

    # Caching with hashable arguments works as usual
    my_slow_function(2)
    my_slow_function(2)
    assert my_slow_function.cache_info().hits == 1  # Used cache

    # Unhashable arguments now also work, but are not memoized
    my_slow_function([2])
    # Warning is emitted that cache is unused
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "Calling unmemoized form" in caplog.text
    # No duplicate warnings
    my_slow_function([2])
    my_slow_function([3])
    assert len(caplog.records) == 1
    assert my_slow_function.cache_info().hits == 1  # Did not use cache

    # Can deactivate warnings
    @nofail_functools_cache(warn=False)
    @cache
    def my_quiet_slow_function(x):
        return x*2

    my_quiet_slow_function([3])
    assert len(caplog.records) == 1

