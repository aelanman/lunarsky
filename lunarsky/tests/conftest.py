import pytest
import warnings

# Ignore a deprecation from spiceypy
@pytest.fixture(autouse=True)
def representation_deprecation_to_error():
    warnings.filterwarnings('ignore', message='Using or importing the ABCs from'
                                              '\'collections\' instead of from'
                                              '\'collections.abc\' is deprecated'
                                    , category=DeprecationWarning)
