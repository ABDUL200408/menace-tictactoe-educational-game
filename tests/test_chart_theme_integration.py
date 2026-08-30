"""Regression checks for Plotly charts following the active Streamlit theme."""

from pathlib import Path


def test_exportable_figures_use_streamlit_theme_without_changing_exports() -> None:
    source = Path("ui/common.py").read_text(encoding="utf-8")

    assert "display_figure = copy.deepcopy(figure)" in source
    assert 'paper_bgcolor="rgba(0,0,0,0)"' in source
    assert 'plot_bgcolor="rgba(0,0,0,0)"' in source
    assert 'theme="streamlit"' in source

    # Export operations must still receive the original report-ready figure.
    assert "export_figure_html(figure, safe_name)" in source
    assert "export_figure_png(figure, safe_name)" in source
