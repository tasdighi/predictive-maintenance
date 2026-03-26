# Predictive Maintenance Anomaly Detection Pipeline

This project is a personal experiment towards as end-to-end data science workflow for predictive maintenance using time-series data. 

The goal is to detect abnormal behavior in industrial assets and support maintenance decisions by building anomaly detection models.

The project follows a CRISP-DM inspired workflow and final goal is to have a reusable Python / PySpark pipeline suitable for production environments.

---

## Motivation

In industrial systems such as conveyor belts, motors, and sensors, large amounts of time-series data are generated continuously.  
Detecting abnormal patterns early can help:

- Reduce downtime
- Optimize asset usage
- Extend component lifetime
- Lower maintenance cost

This project focuses on anomaly detection as a first step toward predictive maintenance.

---

## Project Goals

- Explore time-series sensor data
- Build anomaly detection models using classical ML methods
- Create reusable feature engineering code
- Implement scalable data processing with PySpark
- Design a simple end-to-end ML pipeline
- Using results for maintenance decisions

---

## CRISP-DM Workflow

This project follows the CRISP-DM cycle:

1. Business understanding  
   Detect abnormal behavior to support maintenance decisions

2. Data understanding  
   Explore time-series sensor data

3. Data preparation  
   Feature engineering, cleaning, aggregation

4. Modeling  
   Unsupervised anomaly detection

5. Evaluation  
   Compare models and analyze anomaly scores

6. Deployment considerations  
   Pipeline structure + PySpark + modular code

---

## Methods

### Feature Engineering

Time-series features are created per asset using:

- lag values
- rolling averages
- differences
- statistical features
- window aggregations

Feature engineering is implemented using:

- pandas (prototype)
- PySpark (scalable version)

---

### Anomaly Detection Models

Classical unsupervised anomaly detection methods are used:

- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM


The models output an anomaly score which can be used to create a risk indicator.

---

### Risk Score

risk_score = normalized_anomaly_score + threshold_violation_flag


## Possible Extensions

- failure prediction (classification)
- survival analysis
- deep learning models
- online scoring
- Databricks / cloud deployment
- model monitoring
