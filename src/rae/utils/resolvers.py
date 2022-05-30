from setuptools_scm import get_version

from nn_core.common import PROJECT_ROOT


def codebase_version() -> str:
    version = get_version(root=PROJECT_ROOT)
    return version
