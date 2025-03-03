# Aviation Prediction - FastAPI Backend & Next.js Frontend
![image](https://images.unsplash.com/photo-1623888676435-d3b01b4d1dc0?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

## Link to over video presentation
[Video Presentation](https://drive.google.com/file/d/1H3JEX1B2Ztt5r1WHHNq8P6cLJL5edy5X/view?usp=drive_link)

## Link to Google Drive
- Consists the whole code with charts and datasets.
[Google Drive](https://drive.google.com/drive/folders/1ZwUdBWrEKH_JP1TwYBpcl7lyhVVUnDkF?usp=drive_link)

## Overview

This repository contains a **FastAPI backend** that allows users to train a machine learning model with custom hyperparameters and predict airline profits based on provided data. Additionally, it includes a **Next.js frontend** for user interaction, a **Jupyter Notebook** demonstrating the model's performance using **XGBRegressor**, and an **Exploratory Data Analysis (EDA)** of the dataset.

## Project Background

This project was developed for a **hackathon**, where the challenge was to train a model to predict airline profits. Our primary goal was to develop a **highly accurate machine learning model**. To enhance usability, we also built a web application that enables users to:

- Tune hyperparameters dynamically and visualize their impact.
- Input their own data and obtain profit predictions using a pre-trained model.
- Explore detailed EDA insights and a model performance notebook.

## Tech Stack

- **Backend:** FastAPI (Located in `backend/` folder)
- **Frontend:** Next.js (Located in `dashboard/` folder)
- **Model:** XGBRegressor (Extreme Gradient Boosting Regression)
- **EDA & Model Notebook:** Jupyter Notebook (Located in the root directory)
- **Deployment:** Next.js frontend deployed (Specify hosting if applicable)

## Features

### 1. Machine Learning Model

- Uses **XGBRegressor** for high-accuracy predictions.
- Trained with custom hyperparameters.
- Evaluated using metrics like RMSE and R².
- Results documented in **Model.ipynb**.

### 2. FastAPI Backend

- API endpoints for:
  - Training a model with custom hyperparameters
  - Predicting airline profits based on user-input data
  - Generating visualizations for hyperparameter tuning
- Located in `backend/` folder.

### 3. Next.js Frontend (Dashboard)

- Interactive UI to:
  - Upload datasets
  - Tune hyperparameters
  - View real-time predictions and performance metrics
- Located in `dashboard/` folder.

### 4. Jupyter Notebook - Model & EDA

- **Exploratory Data Analysis (EDA)**:
  - Insights into airline profit data
  - Feature importance analysis
  - Correlation between variables
- **Model Training Notebook**:
  - Training process using **XGBRegressor**
  - Accuracy metrics, RMSE, R² score, etc.
  - Comparisons with baseline models

## Installation & Setup

### Prerequisites

Ensure you have **Python 3.11** and **Node.js** installed.

### Clone the repository

```sh
git clone https://github.com/Bhawesh-Agrawal/NEURAL-NINJAS.git
cd your-repo
```

### Backend Setup

```sh
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
It will only work with python 3.11

Server will be running at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Access the API Documentation

FastAPI provides an interactive API documentation at:

- Swagger UI: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
- Redoc: **[http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)**

### Frontend Setup (Next.js)

```sh
cd dashboard
npm install
npm run dev
```

Front end will be available at [**http://localhost:3000**](http://localhost:3000)

### Running the Jupyter Notebook
Run it in the root folder of the project
```sh
pip install -r requirements.txt
jupyter notebook
```

We have custom function after running all cells you will be promted to choode from 2 option either you can upload a csv file or input a array and the model will predict the output for you.

Open and explore **EDA.ipynb** and **Model**

## API Endpoints

| Method | Endpoint   | Description                                 |
| ------ | ---------- | ------------------------------------------- |
| `POST` | `/train`   | Train the model with custom hyperparameters |
| `POST` | `/predict` | Predict airline profit using input features |            |

## Future Enhancements

- Optimize model hyper parameters for better accuracy
- Deploy the model on a cloud server
- Improve front end UX with better visualizations
- Implement additional ML models for comparison

## Contributors
  *Neural Ninjas*
- Bhawesh Agrawal
- Priyanka Singh
- Shasank Sachan

---


