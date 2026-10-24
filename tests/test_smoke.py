import numpy
import pandas
import pydantic
import pyvista
import qiskit
import qiskit_aer
import scipy
import yaml
from qiskit_aer.primitives import SamplerV2

import lodq


def test_imports() -> None:
    assert lodq.__version__ == "0.1.0"
    assert numpy is not None
    assert scipy is not None
    assert pandas is not None
    assert yaml is not None
    assert pydantic is not None
    assert pyvista is not None
    assert qiskit is not None
    assert qiskit_aer is not None


def test_qiskit_major_version() -> None:
    major = int(qiskit.__version__.split(".", 1)[0])
    assert major >= 2


def test_qiskit_aer_sampler_v2() -> None:
    assert SamplerV2 is not None
