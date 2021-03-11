"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test``.
"""
from pathlib import Path

import pytest

from kedro_impact.run import ProjectContext


@pytest.fixture
def project_context():
    return ProjectContext(str(Path.cwd()))


class TestProjectContext:
    def test_project_name(self, project_context):
        assert project_context.project_name == "kedro-impact"

    def test_project_version(self, project_context):
        assert project_context.project_version == "0.16.5"
