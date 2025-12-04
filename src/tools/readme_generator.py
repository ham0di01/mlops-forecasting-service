"""README generation using Jinja2 templates."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .readme_data import ReadmeData


class ReadmeGenerator:
    """Generates README content using Jinja2 templates."""

    def __init__(self, template_dir: Path = Path("src/tools/templates")):
        self.template_dir = template_dir
        self._setup_jinja_env()

    def _setup_jinja_env(self) -> None:
        """Setup Jinja2 environment with proper configuration."""
        if not self.template_dir.exists():
            raise FileNotFoundError(
                f"Template directory not found: {self.template_dir}"
            )

        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=True,
        )

    def generate_section(
        self, data: ReadmeData, template_name: str = "project_at_a_glance.md.j2"
    ) -> str:
        """Generate a README section from template and data."""
        try:
            template = self.env.get_template(template_name)
            return template.render(
                badges=data.badges,
                metrics=data.metrics,
                timestamp=data.timestamp,
                performance_status=data.performance_status,
                dashboard_link=data.dashboard_link,
                monitoring_content=(
                    data.monitoring_content if data.monitoring_content.strip() else None
                ),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate README section: {e}")


class ReadmeUpdater:
    """Updates README file with new content."""

    SECTION_START = "<!-- PROJECT-AT-A-GLANCE:START -->"
    SECTION_END = "<!-- PROJECT-AT-A-GLANCE:END -->"

    def __init__(self, readme_path: Path = Path("README.md")):
        self.readme_path = readme_path

    def update_section(self, new_section: str) -> None:
        """Update the README with a new section content."""
        if not self.readme_path.exists():
            raise FileNotFoundError(f"README not found: {self.readme_path}")

        try:
            content = self.readme_path.read_text(encoding="utf-8")

            if self.SECTION_START in content and self.SECTION_END in content:
                new_content = self._replace_existing_section(content, new_section)
            else:
                new_content = self._append_new_section(content, new_section)

            self.readme_path.write_text(new_content, encoding="utf-8")
            print("[readme] updated README with Project at a Glance section")

        except Exception as e:
            raise RuntimeError(f"Failed to update README: {e}")

    def _replace_existing_section(self, content: str, new_section: str) -> str:
        """Replace an existing section in the README."""
        pre, _, rest = content.partition(self.SECTION_START)
        _, _, post = rest.partition(self.SECTION_END)
        return pre.rstrip() + "\n" + new_section + post

    def _append_new_section(self, content: str, new_section: str) -> str:
        """Append a new section to the README."""
        return content.rstrip() + "\n\n" + new_section + "\n"
