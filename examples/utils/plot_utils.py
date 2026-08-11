"""Plot utilities used by example notebooks."""

import plotly.express as px
import plotly.graph_objects as go


def histogram(clf, X, nbins=15, fig_type=None):
    """Generate a histogram of positive-class predicted probabilities."""
    y_prob = clf.predict_proba(X)[:, 1]
    fig = px.histogram(y_prob, nbins=nbins)
    fig.update_layout(
        title="Histogram of Predicted Scores",
        xaxis_title="Predicted Scores",
        yaxis_title="Count",
        legend_title="Modelos",
        autosize=False,
        hovermode="x",
        showlegend=False,
    )
    fig.update_traces(hovertemplate="%{y}")
    return fig.show(fig_type)


def plot_prediction_intervals(
    intervals, y_pred, y_test, fig_type=None, width=800, height=400
):
    """Plot prediction intervals, ground truth, and midpoint predictions."""
    fig = go.Figure()

    lower_bound = intervals[:, 0]
    upper_bound = intervals[:, 1]

    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode="lines",
            line=dict(color="darkblue"),
            name="Real Value",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=lower_bound,
            mode="lines",
            line=dict(color="rgba(128, 128, 128, 0.2)"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=upper_bound,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(128, 128, 128, 0.2)",
            line=dict(color="rgba(128, 128, 128, 0.2)"),
            name="Prediction Intervals",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=y_pred,
            mode="lines",
            line=dict(color="red"),
            name="Prediction MidPoint",
        )
    )

    fig.update_layout(
        title="Interval Prediction",
        xaxis_title="Sample",
        yaxis_title="Value",
        legend=dict(title="Metric"),
        width=width,
        height=height,
    )

    return fig.show(fig_type)
