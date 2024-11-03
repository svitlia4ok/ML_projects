This repository contains various machine learning projects, including:

	•	Time Series Forecasting
	•	LSTM Deep Learning for Time Series Prediction

Explore these projects to see implementations of advanced techniques in time series analysis and prediction.

~~~~Time Series Forecasting with ARIMA and Prophet

This notebook is focused on time series forecasting, primarily using the ARIMA and Prophet models.

Table of Contents:

	1.	Data Overview and Preprocessing
	2.	ARIMA Model
	3.	Prophet Model
	4.	Evaluation and Comparison

1. Data Overview and Preprocessing 

	•	Goal: To forecast future values based on historical data using time series models.
	•	Dataset: The data used in this notebook consists of historical time series values, loaded and prepared for modeling.

2. ARIMA Model 

The ARIMA (AutoRegressive Integrated Moving Average) model is commonly used for time series analysis and forecasting. This model is suitable for data that shows patterns such as seasonality and trend.

	•	Parameter Selection: We use the pmdarima library’s auto_arima function to automatically select the optimal parameters (p, d, q) based on model performance metrics like AIC and BIC.
	•	Model Training: After identifying the best parameters, we fit the ARIMA model to the training data.
	•	Forecasting: The model provides forecasts for future time points.
	•	Plotting Results: We visualize the predictions and actual values to evaluate the model’s performance.

3. Prophet Model 

The Prophet model, developed by Facebook, is an additive model known for handling seasonal patterns and holiday effects, making it particularly useful for business and economic data.

	•	Data Preparation: Prophet requires specific column names (ds for dates and y for target values), so the dataset is formatted accordingly.
	•	Model Training: We train the Prophet model on the data and use it for future forecasting.
	•	Plotting Results: The model’s predictions are plotted to visually assess the performance and patterns in forecasted data.

4. Evaluation and Comparison 

	•	Metrics: Both models are evaluated using metrics such as MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error), allowing for direct comparison.
	•	Conclusion: We analyze which model performs best for our dataset, considering both the accuracy metrics and the visual alignment of predictions to actual values.

Libraries Used

	•	pandas and numpy: For data manipulation and handling.
	•	pmdarima: For automatic ARIMA parameter selection.
	•	fbprophet (Prophet): For building and training the Prophet model.
	•	matplotlib and seaborn: For visualizing the data and model predictions.


This notebook provides a comprehensive introduction to time series forecasting using ARIMA and Prophet, with clear steps for data preprocessing, model training, evaluation, and result visualization.




~~~~LSTM Deep Learning for Time Series Prediction

This project utilizes Long Short-Term Memory (LSTM) networks for forecasting time series data, specifically focusing on the prediction of international airline passenger numbers. The goal is to develop a neural network model that can accurately predict future passenger counts based on historical data.

Project Overview

The dataset used in this project spans from January 1949 to December 1960, comprising 12 years of monthly passenger counts. The task is to forecast the number of passengers in thousands for a given month, based on the number of passengers in previous months.

Key Components

	1.	Data Preparation:
	•	The raw passenger count data is preprocessed to create a suitable format for training the LSTM model.
	•	A sliding window approach is used to transform the time series data into input-output pairs for the model.
	2.	LSTM Model Implementation:
	•	A custom class AirModel is defined, inheriting from torch.nn.Module, which includes:
	•	An LSTM layer to capture temporal dependencies in the data.
	•	A linear layer to output the final predictions.
	•	The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function.
	3.	Training Process:
	•	The model is trained over 2000 epochs, with loss recorded and visualized to monitor performance.
	•	Validation steps are incorporated to evaluate the model on unseen data and ensure it generalizes well.
	4.	Results Visualization:
	•	The project includes code to visualize the predicted values against the actual passenger counts, providing insight into the model’s performance on both training and testing datasets.

Conclusion

This project showcases the application of LSTM networks for time series forecasting, demonstrating the effectiveness of deep learning in capturing trends and patterns in sequential data. Through extensive training and validation, the model aims to achieve reliable predictions that can assist in planning and decision-making for airline operations.


~~~~Automated Information Retrieval Using LLMs and Agents

This project aims to develop a system that leverages language models (LLMs) and agents for efficient information retrieval on scientific topics.

	1.	LLM Invocation with Basic Prompt:
The project begins with the invocation of ChatGPT using a simple, structured prompt focused on “Quantum Computing.” The prompt seeks a brief definition, key benefits, and current research developments, limited to 200 characters for efficiency.
	2.	Parameterized Prompt Creation:
Building on the initial setup, the functionality is enhanced to allow dynamic topic input using PromptTemplate from LangChain. This allows the model to generate responses for various topics, including “Bayesian methods in machine learning,” “Transformers in machine learning,” and “Explainable AI.” Each generated explanation is printed for review.
	3.	Agent for Automated Information Retrieval:
The project culminates in creating a ReAct-type agent that automates the search for the latest scientific publications. The agent is guided by a prompt to gather relevant information and produce a structured list of publications, including titles, authors, and brief descriptions.

Overall, this project integrates LLMs and automated agents to streamline the process of acquiring up-to-date information in the scientific field.
