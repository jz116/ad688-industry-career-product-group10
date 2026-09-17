#!/usr/bin/env python3
"""Shared Plotly styling for the Job Market Analysis website.

Every Plotly figure in the project should call ``apply_job_market_theme``
immediately before it is displayed or exported.
"""

from __future__ import annotations

from typing import Literal

import plotly.graph_objects as go


NAVY = "#173B57"
TEAL = "#168A8A"
BLUE = "#4C78A8"
AMBER = "#E29D34"
RED = "#C84C4C"
SLATE = "#667781"
TEXT = "#263640"
GRID = "#E6EDF2"
AXIS = "#C8D0D6"
BACKGROUND = "#FFFFFF"

COLOR_SEQUENCE = [NAVY, TEAL, BLUE, AMBER, RED, "#7A68A6", "#59A14F"]


def apply_job_market_theme(
    fig: go.Figure,
    *,
    title: str,
    subtitle: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    note: str | None = None,
    height: int = 520,
    showlegend: bool = False,
    grid_axis: Literal["x", "y", "both", "none"] = "x",
) -> go.Figure:
    """Apply the project's shared visual design to one Plotly figure.

    Parameters are intentionally chart-neutral so the same function can style
    bar charts, scatter plots, and other figures used across the website.
    """

    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:15px;color:{SLATE}'>{subtitle}</span>"

    bottom_margin = 105 if note else 75

    fig.update_layout(
        template="plotly_white",
        title={
            "text": title_text,
            "x": 0.0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {"family": "Arial, sans-serif", "size": 27, "color": NAVY},
        },
        font={"family": "Arial, sans-serif", "size": 14, "color": TEXT},
        colorway=COLOR_SEQUENCE,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        height=height,
        margin={"l": 90, "r": 70, "t": 115, "b": bottom_margin},
        showlegend=showlegend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1.0,
            "title_text": "",
        },
        hoverlabel={
            "bgcolor": BACKGROUND,
            "bordercolor": AXIS,
            "font": {"family": "Arial, sans-serif", "size": 13, "color": TEXT},
        },
    )

    fig.update_xaxes(
        title_text=x_title or "",
        showgrid=grid_axis in {"x", "both"},
        gridcolor=GRID,
        gridwidth=1,
        showline=True,
        linecolor=AXIS,
        linewidth=1,
        zeroline=False,
        ticks="outside",
        tickcolor=AXIS,
        automargin=True,
        title_font={"size": 16, "color": TEXT},
    )
    fig.update_yaxes(
        title_text=y_title or "",
        showgrid=grid_axis in {"y", "both"},
        gridcolor=GRID,
        gridwidth=1,
        showline=True,
        linecolor=AXIS,
        linewidth=1,
        zeroline=False,
        ticks="outside",
        tickcolor=AXIS,
        automargin=True,
        title_font={"size": 16, "color": TEXT},
    )

    if note:
        fig.add_annotation(
            text=note,
            x=0.0,
            y=-0.20,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
            font={"family": "Arial, sans-serif", "size": 13, "color": SLATE},
        )

    return fig
