import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple

st.set_page_config(page_title="Linear Regression Learning Lab", layout="wide")


def generate_data(a: float, b: float, n_points: int, sigma: float, seed: int, x_min: float, x_max: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(low=x_min, high=x_max, size=n_points)
    true_y = a * x + b
    noise = rng.normal(loc=0.0, scale=sigma, size=n_points)
    y = true_y + noise
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "true_y": true_y,
        "noise": noise,
    })
    return df.sort_values("x").reset_index(drop=True)


def closed_form_solution(df: pd.DataFrame) -> Tuple[float, float, np.ndarray]:
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    intercept, slope = theta.tolist()
    y_pred = X @ theta
    return intercept, slope, y_pred


def sklearn_solution(df: pd.DataFrame) -> Tuple[float, float, np.ndarray]:
    x = df[["x"]]
    y = df["y"]
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    return float(model.intercept_), float(model.coef_[0]), y_pred


def make_scatter_plot(df: pd.DataFrame, a: float, b: float, closed_params: Tuple[float, float], sklearn_params: Tuple[float, float]) -> None:
    true_intercept = b
    true_slope = a
    closed_intercept, closed_slope = closed_params
    sklearn_intercept, sklearn_slope = sklearn_params

    x_vals = df["x"].to_numpy()
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["x"], df["y"], color="#1f77b4", alpha=0.7, label="Synthetic data")
    ax.plot(x_line, true_slope * x_line + true_intercept, color="#2ca02c", label="True line")
    ax.plot(x_line, closed_slope * x_line + closed_intercept, color="#ff7f0e", linestyle="--", label="Normal equation fit")
    ax.plot(x_line, sklearn_slope * x_line + sklearn_intercept, color="#d62728", linestyle=":", label="Scikit-learn fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Scatter, true line, and fitted lines")
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


def make_residual_plot(df: pd.DataFrame, residuals_closed: np.ndarray, residuals_sklearn: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", label="Zero residual")
    ax.scatter(df["x"], residuals_closed, color="#ff7f0e", alpha=0.7, label="Normal equation residuals")
    ax.scatter(df["x"], residuals_sklearn, color="#d62728", alpha=0.7, marker="x", label="Scikit-learn residuals")
    ax.set_xlabel("x")
    ax.set_ylabel("Residual (y - y_hat)")
    ax.set_title("Residual comparison")
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


def crisp_dm_walkthrough() -> None:
    st.subheader("CRISP-DM Walkthrough")
    intro = (
        "This app follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework "
        "to teach linear regression in an interactive way."
    )
    st.write(intro)
    tabs = st.tabs([
        "Business Understanding",
        "Data Understanding",
        "Data Preparation",
        "Modeling",
        "Evaluation",
        "Deployment",
    ])

    with tabs[0]:
        st.write(
            "We aim to explain how simple linear regression fits a straight line to data, "
            "mirroring real-world scenarios where a continuous outcome depends on a single feature."
        )
    with tabs[1]:
        st.write(
            "Use the controls to inspect how slope, intercept, noise, and sample size change the synthetic data. "
            "Observe the scatter plot to understand trends and variance."
        )
    with tabs[2]:
        st.write(
            "Data generation doubles as preparation: you create a clean dataset with known ground truth, "
            "making it easier to reason about bias, variance, and residual behavior."
        )
    with tabs[3]:
        st.write(
            "We train two models: the analytical normal equation solution and scikit-learn's implementation. "
            "Comparing them highlights how tooling replicates closed-form math."
        )
    with tabs[4]:
        st.write(
            "Key metrics (MSE, MAE, and R^2) quantify model performance. Residual plots reveal systematic patterns."
        )
    with tabs[5]:
        st.write(
            "Download the data and deploy the app with one click (see README) to turn this learning artifact into a shareable demo."
        )


def main() -> None:
    st.title("Linear Regression Learning Lab")
    st.caption("Explore simple linear regression end-to-end through the CRISP-DM lens.")

    with st.sidebar:
        st.header("Data Generation")
        a = st.slider("Slope (a)", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
        b = st.slider("Intercept (b)", min_value=-10.0, max_value=10.0, value=1.0, step=0.5)
        sigma = st.slider("Noise (sigma)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        n_points = st.slider("Number of points (N)", min_value=20, max_value=1000, value=200, step=10)
        x_min, x_max = st.slider("X range", min_value=-25.0, max_value=25.0, value=(-10.0, 10.0), step=0.5)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
        st.caption("All randomness comes from the seed to keep runs reproducible.")

    data = generate_data(a, b, n_points, sigma, seed, x_min, x_max)

    closed_intercept, closed_slope, closed_preds = closed_form_solution(data)
    sklearn_intercept, sklearn_slope, sklearn_preds = sklearn_solution(data)

    residuals_closed = data["y"].to_numpy() - closed_preds
    residuals_sklearn = data["y"].to_numpy() - sklearn_preds

    metrics_df = pd.DataFrame(
        {
            "Normal Equation": [
                mean_squared_error(data["y"], closed_preds),
                mean_absolute_error(data["y"], closed_preds),
                r2_score(data["y"], closed_preds),
            ],
            "Scikit-learn": [
                mean_squared_error(data["y"], sklearn_preds),
                mean_absolute_error(data["y"], sklearn_preds),
                r2_score(data["y"], sklearn_preds),
            ],
        },
        index=["MSE", "MAE", "R^2"],
    )

    crisp_dm_walkthrough()

    st.markdown("---")
    st.subheader("Model Fits")
    col1, col2 = st.columns([2, 1])
    with col1:
        make_scatter_plot(data, a, b, (closed_intercept, closed_slope), (sklearn_intercept, sklearn_slope))
    with col2:
        st.markdown("**Parameter recap**")
        st.write(
            pd.DataFrame(
                {
                    "Model": ["True", "Normal equation", "Scikit-learn"],
                    "Intercept": [b, closed_intercept, sklearn_intercept],
                    "Slope": [a, closed_slope, sklearn_slope],
                }
            ).set_index("Model")
        )

    st.subheader("Residual Diagnostics")
    make_residual_plot(data, residuals_closed, residuals_sklearn)

    st.subheader("Performance Metrics")
    st.dataframe(metrics_df.style.format("{:.4f}"))

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10))
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download synthetic dataset (CSV)", data=csv, file_name="linear_regression_synthetic.csv", mime="text/csv")

    st.info(
        "Tip: Adjust the sliders to see how noise, sample size, and true coefficients change the fit and metrics."
    )


if __name__ == "__main__":
    main()
