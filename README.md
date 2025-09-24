# Linear Regression Learning Lab

Interactive Streamlit app that walks through simple linear regression end-to-end using the CRISP-DM framework.

## Live Demo
- Live demo on Streamlit Cloud: https://aiotda.streamlit.app/7114056076

Generate synthetic data with a known line, compare closed-form and scikit-learn fits, inspect diagnostics, and export the dataset for downstream exploration.

## Features
- Synthetic data from the equation `y = a * x + b + noise` with user controls for slope, intercept, sample size, noise, x-range, and random seed.
- Deterministic data generation by reusing the provided seed for every random draw.
- Side-by-side comparison between the normal equation solution and scikit-learn's `LinearRegression`.
- Visual diagnostics: scatter plot with true and fitted lines plus residual plot.
- Metrics dashboard with MSE, MAE, and R^2 for each model.
- Preview table and CSV download for the generated dataset.

## CRISP-DM Narrative
- **Business Understanding**: Explain how linear regression models a one-feature prediction task and why the comparison between theory and tooling matters to practitioners.
- **Data Understanding**: Adjust the sliders to study how slope, intercept, noise, and sample size influence the scatter, variance, and trend.
- **Data Preparation**: Generate a clean dataset with a known ground truth line. Sorting by `x` and exposing the noise column keep the data tidy and explainable.
- **Modeling**: Fit the analytical closed-form (normal equation) solution alongside scikit-learn's `LinearRegression` for an apples-to-apples comparison.
- **Evaluation**: Review the metrics table and residual plot to spot systematic errors or variance that the line misses.
- **Deployment**: Share the interactive experience as a Streamlit Community Cloud app and export the CSV for reuse in notebooks or slide decks.

## Run Locally
1. (Optional) create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Launch the app: `streamlit run app.py`.
4. Visit the provided local URL in your browser.

## One-Click Deployment (Streamlit Community Cloud)
1. Push this project to a GitHub repository that includes `app.py` and `requirements.txt`.
2. Sign in at [https://streamlit.io/cloud](https://streamlit.io/cloud) and click **New app**.
3. Select the repository, choose the branch, and set `app.py` as the entry point.
4. Confirm that the dependency file is `requirements.txt`, then click **Deploy**. Streamlit builds and hosts the app automatically.
5. Share the generated URL. Future pushes to the branch trigger automatic redeploys.

## Tips for Exploration
- Toggle the random seed to reproduce or contrast experiments without changing other settings.
- Increase the noise term to see how residual spread affects MSE, MAE, and R^2.
- Increase the number of points to watch both estimators converge to the true slope and intercept.

## Development Summary
This section provides a summary of the development steps taken, as recorded in 0_devLog.md.

1.0-2.0: Initial setup, including the creation of the development log (0_devLog.md) and the to-do list (Todo.md).
3.0-4.0: Modification and verification of the project plan.
5.0: Execution of the project plan, starting with the creation of the Streamlit application (app.py).
6.0-11.0: Troubleshooting and resolving issues related to running the Streamlit application, including the creation of requirements.txt, setting up a virtual environment, and installing dependencies.
12.0-13.0: Successfully running the Streamlit application.


