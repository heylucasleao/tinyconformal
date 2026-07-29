# Copyright (c) 2024-2026 Lucas Leão
# tinyCP - A small toolbox for conformal prediction
# Licensed under the MIT License


import numpy as np
from typing import List
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from scipy.stats import beta


def efficiency_curve(
    clf, X: np.ndarray, y: np.ndarray, fig_type=None, width=800, height=400
):
    """
    Compute and plot the efficiency and validity curves for a conformal classifier.

    Evaluates the model's statistical calibration (validity) and empirical
    usefulness (efficiency) across a predefined range of significance levels
    (alpha). This visualization helps diagnose under- or over-conservatism and
    identifies the optimal operating alpha for the classifier.

    Parameters
    ----------
    clf : object
        The conformal classifier model. Must implement a `predict_set(X, alpha)`
        method that returns a binary matrix of prediction sets.
    X : array-like of shape (n_samples, n_features)
        Input data features used for evaluation.
    y : array-like of shape (n_samples,) or (n_samples, 1)
        True target labels for the input data.
    fig_type : str, optional
        Format to automatically save or render the figure (e.g., 'png', 'svg').
        If None, the interactive plot is displayed in the browser.
    width : int, default=800
        The width of the generated Plotly figure in pixels.
    height : int, default=400
        The height of the generated Plotly figure in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        An interactive Plotly figure object containing the Validity, Efficiency,
        and Theoretical Coverage curves.

    Raises
    ------
    AttributeError
        If the provided `clf` object does not have a `predict_set` method.
    ValueError
        If the shapes of `X` and `y` are incompatible.

    Notes
    -----
    - **Validity** measures empirical coverage, defined as the proportion of
    samples where the true label is included in the prediction set. A valid
    model stays on or above the :math:`1 - \alpha` line.
    - **Efficiency** is quantified here as the *singleton rate* (the fraction of
    prediction sets containing exactly one class). Higher values indicate more
    informative and precise predictions.
    - A drop in efficiency at high alpha levels typically indicates the generation
    of empty prediction sets, as the model drops classes to meet the high
    permitted error budget.

    References
    ----------
    .. Angelopoulos, A. N., & Replinger, S. (2021). A gentle introduction to
    conformal prediction and distribution-free uncertainty quantification.
    arXiv preprint arXiv:2107.07511.
    .. Shafer, G., & Vovk, V. (2008). A tutorial on conformal prediction.
    Journal of Machine Learning Research, 9(3).
    """

    def get_error_metrics(clf, X: np.ndarray, y: np.ndarray) -> tuple:
        error_rate = np.asarray(
            [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01]
        )
        efficiency_rate = np.zeros(error_rate.shape)
        validity_rate = np.zeros(error_rate.shape)

        y_flat = y.flatten().astype(int)
        n_samples = len(X)

        for i, error in enumerate(error_rate):
            predict_set = clf.predict_set(X, alpha=error)
            set_sizes = predict_set.sum(axis=1)
            efficiency_rate[i] = np.sum(set_sizes == 1) / n_samples
            covered = predict_set[np.arange(n_samples), y_flat]
            validity_rate[i] = np.mean(covered)

        return error_rate, efficiency_rate, validity_rate

    error_rate, efficiency_rate, validity_rate = get_error_metrics(clf, X, y)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=error_rate,
            y=efficiency_rate,
            mode="lines+markers",
            name="Efficiency (Singleton Rate)",
            line=dict(color="darkblue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=error_rate,
            y=validity_rate,
            mode="lines+markers",
            name="Validity (Empirical Coverage)",
            line=dict(color="orange"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=error_rate,
            y=1 - error_rate,
            mode="lines",
            name="Theoretical Coverage (1 - alpha)",
            line=dict(color="grey", dash="dash"),
        )
    )

    fig.update_layout(
        title="Efficiency & Validity Curve",
        xaxis_title="Significance Level (Alpha / Error Rate)",
        yaxis_title="Score / Rate",
        legend=dict(title="Metric"),
        width=width,
        height=height,
        hovermode="x",
    )
    fig.update_traces(hovertemplate="%{y:.2f}")

    return fig.show(fig_type)


def reliability_curve(
    clf, X, y, n_bins=15, fig_type=None, model_name="RandomForest"
) -> go.Figure:
    """
    Generates a reliability curve for a classifier.

    Args:
        clf (object): The classifier model.
        X (np.ndarray): Input data.
        y (np.ndarray): True labels.
        n_bins (int, optional): Number of bins for the reliability curve. Defaults to 15.
        fig_type (str, optional): Type of figure to display (e.g., 'png', 'svg'). Defaults to None.

    Returns:
        go.Figure: Reliability curve plot.
    """

    y_prob = clf.predict_proba(X)[:, 1]

    v_prob_true, v_prob_pred = calibration_curve(
        y, y_prob, n_bins=n_bins, strategy="quantile"
    )

    fig = go.Figure()

    # Add traces for each model

    fig.add_trace(
        go.Scatter(x=v_prob_pred, y=v_prob_true, mode="lines+markers", name=model_name)
    )

    # Add a trace for the perfectly calibrated line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfectly calibrated",
            line=dict(dash="dash", color="grey"),
        )
    )

    fig.update_layout(
        title="Reliability Curve",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        legend_title="Model",
        autosize=False,
    )

    return fig.show(fig_type)


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


def confusion_matrix(clf, X, y, alpha=None, fig_type=None, percentage_by_class=True):
    """
    Generates an annotated heatmap of the confusion matrix for a classifier.

    Args:
        clf: Classifier object (e.g., sklearn classifier).
        X: Input features.
        y: True labels.
        alpha: Optional parameter for classifier prediction.
        fig_type: Optional figure type (e.g., 'png', 'svg').
        percentage_by_class: If True, displays percentages by class; otherwise, overall percentages.

    Returns:
        Annotated heatmap of the confusion matrix.
    """

    y_pred = clf.predict(X, alpha)
    tn, fp, fn, tp = sklearn_confusion_matrix(y, y_pred).ravel()
    labels = np.array([["FN", "TN"], ["TP", "FP"]])
    cm = np.array([[fn, tn], [tp, fp]])

    if percentage_by_class:
        total = cm.sum(axis=0)
        percentage = cm / total * 100
    else:
        percentage = cm / np.sum(cm) * 100

    annotation_text = np.empty_like(percentage, dtype="U10")

    for i in range(percentage.shape[0]):
        for j in range(percentage.shape[1]):
            annotation_text[i, j] = f"{labels[i, j]} {percentage[i, j]:.2f}"

    fig = ff.create_annotated_heatmap(
        cm,
        x=["Positive", "Negative"],
        y=["Negative", "Positive"],
        colorscale="Blues",
        hoverinfo="z",
        annotation_text=annotation_text,
    )

    fig.update_layout(width=400, height=400, title="Confusion Matrix")
    return fig.show(fig_type)


def beta_pdf_with_cdf_fill(alpha, beta_param, fig_type=None, start=0, end=1.0):
    """
    Plot the Beta Probability Density Function (PDF) with an optional fill between a specified interval,
    and display the cumulative density from the CDF as text.

    Parameters:
    alpha (int or float): The alpha (α) parameter of the Beta distribution.
    beta_param (int or float): The beta (β) parameter of the Beta distribution.
    fig_type: Optional figure type (e.g., 'png', 'svg').
    start (float): The starting value of the interval to fill. Default is 0.
    end (float): The ending value of the interval to fill. Default is 1.0.

    Returns:
    A Plotly figure displaying the Beta PDF with the specified filled interval and annotated cumulative density.
    """

    x = np.linspace(0, 1, 1000)
    y_pdf = beta.pdf(x, alpha, beta_param)

    fill_indices = (x >= start) & (x <= end)
    x_fill = x[fill_indices]
    y_pdf_fill = y_pdf[fill_indices]

    cumulative_density = beta.cdf(end, alpha, beta_param) - beta.cdf(
        start, alpha, beta_param
    )

    trace_pdf = go.Scatter(
        x=x, y=y_pdf, mode="lines", name=f"Beta PDF(α={alpha}, β={beta_param})"
    )
    trace_fill = go.Scatter(
        x=x_fill,
        y=y_pdf_fill,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.2)",
        name=f"Interval [{start}, {end}]",
    )

    layout = go.Layout(
        title="Beta PDF with CDF Fill",
        xaxis=dict(title="Success Rate", range=[min(x_fill) - 0.1, 1]),
        yaxis=dict(title="Density"),
        annotations=[
            dict(
                x=(start + end) / 2 if len(y_pdf_fill) > 0 else start * 1.1,
                y=max(y_pdf_fill) * 1.1,
                xref="x",
                yref="y",
                text=f"Cumulative Density: {cumulative_density:.2f}",
                showarrow=False,
                opacity=0.8,
                align="center",
            )
        ],
        width=800,
        height=400,
    )

    fig = go.Figure(data=[trace_pdf, trace_fill], layout=layout)

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
