## WQU MScFE 600 Financial Data — Group Work Project 1
A comprehensive quantitative finance pipeline developed for WorldQuant University (WQU) MScFE 600 (Financial Data) Course. This repository contains the mathematical modeling frameworks, structural analysis scripts, and empirical asset pipelines required to complete all core modules of Group Work Project #1.

## Contributors - ME & SHIJIE DING
------------------------------
## 🚀 Key Modules & Analytical Scope
## 1. Data Quality & Structural Integrity Assessment

* Structured Data Quality: Evaluates structural failures in relational financial tables, tracking anomalies like null entries, physical impossibilities (negative volume), domain boundary violations, and non-unique records.
* Unstructured Data Diagnostics: Assesses unparsed natural language data (analyst sentiment scrapes) for precision failures, excessive semantic noise, temporal drift, and speculative ambiguity.

## 2. Yield Curve Modeling

* Nelson-Siegel Parametric Fit: Fits an ordinary least squares optimization model to sovereign interest rate maturities to extract smooth macroeconomic risk parameters ($\beta_0$ Level, $\beta_1$ Slope, $\beta_2$ Curvature).
* Non-Parametric Cubic Spline: Implements piecewise third-degree polynomial interpolation using continuous boundary conditions at regional knots.
* Comparative Evaluation: Measures performance across the global smoothing profile vs. localized interpolation limits and examines the ethics of parametric data filtering.

## 4. Empirical Sector ETF Decomposition

* Asset Asset Ingestion: Extracts historical closing daily price series for the top 30 structural holdings of a targeted sector ETF over a minimum 6-month tracking window.
* Returns Pipeline: Transforms non-stationary nominal asset valuations into stationary, time-additive log returns.
* Matrix Factorizations: Runs parallel matrix transformations using Principal Component Analysis (PCA) via the covariance matrix and direct Singular Value Decomposition (SVD) on the scaled data matrix.
* Equivalence Verification: Mathematically verifies that $\lambda_i = s_i^2$, confirming that direct data matrix factoring provides identical results to asset covariance eigendecomposition with superior numerical stability.

------------------------------
## 🛠️ Quantitative Tech Stack

* Language: Python 3.10+
* Core Calculus & Statistics: numpy, scipy (specifically scipy.optimize and scipy.linalg)
* Data Manipulation & Preprocessing: pandas
* Linear Algebra & Matrix Transformations: scikit-learn (PCA modeling engine)
* Visualizations & Diagnostics: matplotlib (Scree plot and yield curve curves)
* Environment: Google Colab / Jupyter Notebooks (.ipynb)

------------------------------
## ⚙️ Running the Pipeline Locally

   1. Clone this repository to your local system:
   
   git clone https://github.com
   cd WQU-MScFE600-GWP1
   
   2. Establish an isolated virtual environment and run the system deployment file:
   
   python -m venv venv
   source venv/bin/activate  # For Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   
   3. Open the main program loop inside your Jupyter interface:
   
   jupyter notebook notebook/financial_data_gwp1.ipynb
   
   
------------------------------
## ⚖️ Academic Integrity Note
This code base is submitted in compliance with the WorldQuant University Academic Policy. It is strictly intended as an educational reference architecture for student groups working through the MScFE 600 curriculum. Reusing or re-submitting these sections verbatim violates program policies and triggers Turnitin similarity flags.
------------------------------
