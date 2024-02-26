# Skye-Jiajun MLOps Project

## Overview

This project aims to develop a web application for predicting rental home prices and classifying mushrooms. It utilizes machine learning models trained on relevant datasets to provide accurate predictions and classifications.

## Folder Structure

- **app.py:** Main Flask application file
- **requirements.txt:** List of dependencies for the project
- **runtime.txt:** Specifies Python runtime version for Heroku
- **Procfile:** Heroku configuration file
- **README.md:** Project README file
- **data/:** Directory for storing dataset files
  - `01_homely_resort_listing.csv`: Dataset containing resort listings
- **models/:** Directory for storing trained ML models
  - **jiajun/:** Model trained by your friend
    - `final_model_mushroom.pkl`
  - **skye/:** Your own trained model
    - `tuned_model_for_price_prediction.pkl`
- **templates/:** Flask templates for web pages
  - `jiajun_classification.html`: Web page for Jia Jun's mushroom classification
  - `skye_prediction.html`: Web page for predicting rental home prices
- **.gitignore:** Git ignore file to exclude certain files and directories

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/skye76439/skye-jiajun-mlops.git
   cd skye-jiajun-mlops



# Tasks Overview

## Task 1: Exploratory Data Analysis (EDA)

- Conducted exploratory data analysis on the homely resort listing dataset (`data/01_homely_resort_listing.csv`).
- Explored various features such as accommodations, bathrooms, bedrooms, beds, availability, host characteristics, and review scores.
- Investigated data distributions, correlations, and any potential patterns or outliers.

## Task 2: Model Training with PyCaret

- Utilized PyCaret for automated machine learning to train a predictive model for rental home prices.
- Preprocessed the data by handling missing values, encoding categorical variables, and scaling numerical features.
- Explored different regression models provided by PyCaret and selected the best-performing model.
- Evaluated the model's performance using cross-validation and assessed key metrics such as RMSE and R-squared.

## Task 3: Web Application Development

- Developed a Flask-based web application (`app.py`) to provide a user interface for predicting rental home prices and mushroom classification.
- Created separate HTML templates (`jiajun_classification.html` and `skye_prediction.html`) for each prediction task.
- Integrated the trained machine learning models (`final_model_mushroom.pkl` and `tuned_model_for_price_prediction.pkl`) into the web application for making predictions.
- Implemented navigation between prediction pages using a navbar to switch between Jia Jun's mushroom classification and Skye's rental home price prediction.

## Task 4: Project Enhancement with Poetry, Hydra, and DVC

- Introduced Poetry for managing project dependencies and creating a standardized project environment (`pyproject.toml` and `poetry.lock`).
- Incorporated Hydra for more flexible configuration management, allowing easy adjustment of model parameters and other settings (`configs/config.yaml`).