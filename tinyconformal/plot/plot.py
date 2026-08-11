# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


import numpy as np
from typing import List
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from scipy.stats import beta


def histogram(clf, X, nbins=15, fig_type=None):
    """
    Generates a histogram of predicted scores for a classifier.

    Args:
        clf (object): The classifier model.
        X (np.ndarray): Input data.
        nbins (int, optional): Number of bins for the histogram. Defaults to 15.
        fig_type (str, optional): Type of figure to display (e.g., 'png', 'svg'). Defaults to None.

    Returns:
        A histogram plot.
    """
    y_prob = clf.predict_proba(X)[:, 1]
    fig = px.histogram(y_prob, nbins=nbins)
    fig.update_layout(
        title="Histogram of Predicted Scores",
        xaxis_title="Predicted Scores",
        yaxis_title="Count",
        legend_title="Modelos",
        autosize=False,
    )
    fig.update_layout(hovermode="x")
    fig.update_traces(hovertemplate="%{y}")
    fig.update_layout(showlegend=False)
    return fig.show(fig_type)


def plot_prediction_intervals(
    intervals, y_pred, y_test, fig_type=None, width=800, height=400
):
    """
    Generates an interactive plot to visualize prediction intervals,
    actual values, and model predictions.

    Parameters:
    - intervals (numpy.ndarray): Array containing the lower and upper bounds of the prediction intervals.
    - y_pred (numpy.ndarray): Array containing the predicted values from the model.
    - y_test (pandas.Series): Series containing the actual test set values.

    Returns:
    - fig (plotly.graph_objects.Figure): Interactive Plotly figure object.
    """
    fig = go.Figure()

    lower_bound = intervals[:, 0]
    upper_bound = intervals[:, 1]

    # Valores reais
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode="lines",
            line=dict(color="darkblue"),
            name="Real Value",
        )
    )

    # Limite inferior
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=lower_bound,
            mode="lines",
            line=dict(color="rgba(128, 128, 128, 0.2)"),
            showlegend=False,
        )
    )

    # Limite superior
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=upper_bound,
            mode="lines",
            fill="tonexty",  # Preenche entre este trace e o anterior
            fillcolor="rgba(128, 128, 128, 0.2)",
            line=dict(color="rgba(128, 128, 128, 0.2)"),
            name="Prediction Intervals",
        )
    )

    # Previsões
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
