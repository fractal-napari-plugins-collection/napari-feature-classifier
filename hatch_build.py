"""Build-time metadata hook for the napari-feature-classifier plugin.

The plugin and its headless companion ``feature-classifier-core`` are versioned
in lockstep from the same repository git tag. We therefore pin the core to the
*exact* same version as the plugin being built.
"""

from __future__ import annotations

from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata: dict) -> None:
        version = _resolve_version(Path(self.root))
        core = (
            f"feature-classifier-core=={version}"
            if version
            else "feature-classifier-core"
        )
        metadata["dependencies"] = [core, *self.config["base-dependencies"]]


def _resolve_version(root: Path) -> str | None:
    """Resolve the version from git, falling back to the written _version.py.

    During a normal build the git tag is available. When building a wheel from
    an sdist (no ``.git``), hatch-vcs has already written ``_version.py`` into
    the tree, so we read it from there.
    """
    try:
        from setuptools_scm import get_version

        return get_version(root=str(root), relative_to=str(root / "hatch_build.py"))
    except Exception:
        version_file = root / "src" / "napari_feature_classifier" / "_version.py"
        try:
            namespace: dict = {}
            exec(version_file.read_text(), namespace)  # noqa: S102
            return namespace.get("__version__")
        except Exception:
            return None
