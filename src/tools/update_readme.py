"""Elegant README updater using Jinja2 templates and clean architecture."""

from __future__ import annotations

from .readme_data import ReadmeDataCollector
from .readme_generator import ReadmeGenerator, ReadmeUpdater


def main() -> int:
    """Update README with latest project metrics and information."""
    try:
        # Collect all necessary data
        collector = ReadmeDataCollector()
        data = collector.collect()

        # Generate the section using Jinja2 template
        generator = ReadmeGenerator()
        section = generator.generate_section(data)

        # Update the README file
        updater = ReadmeUpdater()
        updater.update_section(section)

        return 0

    except Exception as e:
        print(f"[readme] Error updating README: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
