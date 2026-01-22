"""
Tests for Step 1: Project Setup

Verifies:
- Project structure is correct
- All modules can be imported
- Version information is correct
- CLI is properly configured
- Placeholder functions are in place
"""

from pathlib import Path

import pytest


class TestProjectStructure:
    """Test that all required files and directories exist."""

    def test_src_directory_exists(self):
        """Verify src/ directory exists."""
        src_path = Path(__file__).parent.parent / "src"
        assert src_path.exists(), "src/ directory should exist"
        assert src_path.is_dir(), "src/ should be a directory"

    def test_tests_directory_exists(self):
        """Verify tests/ directory exists."""
        tests_path = Path(__file__).parent
        assert tests_path.exists(), "tests/ directory should exist"
        assert tests_path.is_dir(), "tests/ should be a directory"

    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml exists."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"

    def test_all_source_modules_exist(self):
        """Verify all required source modules exist."""
        src_path = Path(__file__).parent.parent / "src"
        expected_modules = [
            "__init__.py",
            "main.py",
            "youtube_client.py",
            "audio_analyzer.py",
            "similarity.py",
            "explainer.py",
        ]
        for module in expected_modules:
            module_path = src_path / module
            assert module_path.exists(), f"{module} should exist in src/"


class TestModuleImports:
    """Test that all modules can be imported."""

    def test_import_src_package(self):
        """Verify src package can be imported."""
        import src

        assert hasattr(src, "__version__")

    def test_import_main(self):
        """Verify main module can be imported."""
        from src import main

        assert hasattr(main, "cli")
        assert hasattr(main, "search")

    def test_import_youtube_client(self):
        """Verify youtube_client module can be imported."""
        from src import youtube_client

        assert hasattr(youtube_client, "search_track")
        assert hasattr(youtube_client, "get_audio")
        assert hasattr(youtube_client, "search_candidates")
        assert hasattr(youtube_client, "get_related_videos")

    def test_import_audio_analyzer(self):
        """Verify audio_analyzer module can be imported."""
        from src import audio_analyzer

        assert hasattr(audio_analyzer, "analyze_melody")
        assert hasattr(audio_analyzer, "analyze_rhythm")
        assert hasattr(audio_analyzer, "analyze_track")

    def test_import_similarity(self):
        """Verify similarity module can be imported."""
        from src import similarity

        assert hasattr(similarity, "build_faiss_index")
        assert hasattr(similarity, "find_similar")

    def test_import_explainer(self):
        """Verify explainer module can be imported."""
        from src import explainer

        assert hasattr(explainer, "generate_reasoning")


class TestVersionInfo:
    """Test version information is correct."""

    def test_package_version(self):
        """Verify package version is set correctly."""
        import src

        assert src.__version__ == "0.1.0"


class TestCLISetup:
    """Test CLI is properly configured with Click."""

    def test_cli_group_exists(self):
        """Verify CLI group is defined."""
        from src.main import cli

        assert cli is not None
        # Click groups are callable
        assert callable(cli)

    def test_search_command_exists(self):
        """Verify search command is defined."""
        from src.main import search

        assert search is not None
        assert callable(search)

    def test_cli_invocation(self):
        """Test CLI can be invoked with --help."""
        from click.testing import CliRunner

        from src.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "CLI Music Similarity Tool" in result.output

    def test_cli_version(self):
        """Test CLI shows version."""
        from click.testing import CliRunner

        from src.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_search_command_help(self):
        """Test search command has help text."""
        from click.testing import CliRunner

        from src.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0
        assert "QUERY" in result.output
        assert "--type" in result.output
        assert "--limit" in result.output
        assert "--threshold" in result.output

    def test_search_command_options(self):
        """Test search command accepts all options."""
        from click.testing import CliRunner

        from src.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli, ["search", "test query", "--type", "melody", "--limit", "5"]
        )
        # Should work (placeholder output)
        assert result.exit_code == 0
        assert "test query" in result.output
        assert "melody" in result.output

    def test_search_type_choices(self):
        """Test search --type only accepts valid choices."""
        from click.testing import CliRunner

        from src.main import cli

        runner = CliRunner()

        # Valid types should work
        for valid_type in ["melody", "rhythm", "both"]:
            result = runner.invoke(cli, ["search", "test", "--type", valid_type])
            assert result.exit_code == 0

        # Invalid type should fail
        result = runner.invoke(cli, ["search", "test", "--type", "invalid"])
        assert result.exit_code != 0


class TestPlaceholderFunctions:
    """Test that placeholder functions raise NotImplementedError.

    Note: YouTube functions are now implemented and tested in test_youtube_client.py
    Note: Audio analyzer functions are now implemented and tested in test_audio_analyzer.py
    """

    def test_build_faiss_index_not_implemented(self):
        """Verify build_faiss_index raises NotImplementedError."""
        import numpy as np

        from src.similarity import build_faiss_index

        with pytest.raises(NotImplementedError) as exc_info:
            build_faiss_index(np.array([[1, 2, 3]]))
        assert "Step 4" in str(exc_info.value)

    def test_find_similar_not_implemented(self):
        """Verify find_similar raises NotImplementedError."""
        import numpy as np

        from src.similarity import find_similar

        with pytest.raises(NotImplementedError) as exc_info:
            find_similar(np.array([1, 2, 3]), np.array([[1, 2, 3]]))
        assert "Step 4" in str(exc_info.value)

    def test_generate_reasoning_not_implemented(self):
        """Verify generate_reasoning raises NotImplementedError."""
        from src.explainer import generate_reasoning

        with pytest.raises(NotImplementedError) as exc_info:
            generate_reasoning({}, {})
        assert "Step 5" in str(exc_info.value)


class TestPyprojectToml:
    """Test pyproject.toml is properly configured."""

    def test_pyproject_contains_dependencies(self):
        """Verify pyproject.toml contains all required dependencies."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()

        required_deps = [
            "yt-dlp",
            "essentia",
            "faiss-cpu",
            "click",
            "python-dotenv",
            "requests",
            "numpy",
        ]

        for dep in required_deps:
            assert dep in content, f"pyproject.toml should contain {dep}"

    def test_pyproject_contains_dev_dependencies(self):
        """Verify pyproject.toml contains dev dependencies."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()

        dev_deps = ["pytest", "pytest-cov"]

        for dep in dev_deps:
            assert dep in content, f"pyproject.toml should contain {dep}"

    def test_pyproject_contains_cli_script(self):
        """Verify pyproject.toml defines CLI entry point."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()

        assert "lyrebird" in content
        assert "src.main:cli" in content
