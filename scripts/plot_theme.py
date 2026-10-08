PLOTLY_COLORS = [
    "#2563EB",  # blue
    "#0F766E",  # green
    "#D97706",  # orange
    "#7C3AED",  # purple
    "#DC2626"   # red
]


def apply_plotly_theme(
    fig,
    title,
    x_title=None,
    y_title=None,
    height=500,
    show_legend=True
):
    fig.update_layout(
        template="plotly_white",
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 22}
        },
        font={
            "family": "Arial, sans-serif",
            "size": 13,
            "color": "#1F2937"
        },
        colorway=PLOTLY_COLORS,
        height=height,
        showlegend=show_legend,
        legend={"title_text": ""},
        margin={"l": 70, "r": 30, "t": 85, "b": 70},
        hoverlabel={
            "bgcolor": "white",
            "font_size": 12
        },
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    fig.update_xaxes(
        showline=True,
        linecolor="#D1D5DB",
        gridcolor="#E5E7EB",
        zeroline=False
    )

    fig.update_yaxes(
        showline=True,
        linecolor="#D1D5DB",
        gridcolor="#E5E7EB",
        zeroline=False
    )

    if x_title is not None:
        fig.update_xaxes(title_text=x_title)

    if y_title is not None:
        fig.update_yaxes(title_text=y_title)

    return fig
