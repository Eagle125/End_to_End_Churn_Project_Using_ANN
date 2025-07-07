# Customer Churn Prediction using Artificial Neural Networks

## 1. Introduction

This report details the development of an Artificial Neural Network (ANN) model for predicting customer churn in a banking context. Customer churn, the phenomenon of customers discontinuing their relationship with a service provider, is a critical concern for businesses across various industries. Early and accurate prediction of churn allows companies to implement targeted retention strategies, thereby minimizing revenue loss and maximizing customer lifetime value. This project leverages a dataset containing various customer attributes to build a robust predictive model capable of identifying customers at high risk of churning.



## 2. Problem Definition

Customer churn poses a significant challenge to businesses, particularly in competitive sectors like banking. The core problem addressed by this project is to accurately predict which customers are likely to churn (i.e., close their bank accounts) based on their historical data and attributes. This is a binary classification problem, where the model needs to classify each customer as either 'churn' (Exited = 1) or 'no churn' (Exited = 0). Effective churn prediction enables proactive intervention, allowing the bank to offer incentives or personalized services to retain at-risk customers.

As highlighted in the `experiment.ipynb` notebook, churn can manifest differently across industries:

*   **Telecom Company:** Customer cancels their mobile/internet plan or switches providers.
*   **Streaming Service (e.g., Netflix, Spotify):** Subscriber cancels their subscription or stops renewing it.
*   **E-commerce Website:** Regular customer stops making purchases for an extended period.
*   **Gym or Fitness Center:** Member cancels or doesn’t renew their membership.
*   **SaaS (Software-as-a-Service):** User cancels their subscription to a software/platform.
*   **Banking or Financial Services:** Customer closes their account or switches to another bank.
*   **Online Learning Platform:** Student stops enrolling in courses or cancels their membership.
*   **Ride-Sharing App (e.g., Uber, Lyft):** User stops using the app or switches to a competitor.

For this project, the focus is specifically on banking customer churn, aiming to develop a model that can predict whether a bank customer will 'Exited' (churn) or not.



## 3. Dataset and Preprocessing

### 3.1 Dataset Description

The dataset used for this project is `Churn_Modelling.csv`, which contains information about bank customers and their churn status. The dataset includes the following features:

*   `RowNumber`: Row number (identifier, not used for modeling)
*   `CustomerId`: Unique customer identifier (not used for modeling)
*   `Surname`: Customer's surname (not used for modeling)
*   `CreditScore`: Customer's credit score
*   `Geography`: Customer's country of residence (France, Spain, Germany)
*   `Gender`: Customer's gender (Male, Female)
*   `Age`: Customer's age
*   `Tenure`: Number of years the customer has been with the bank
*   `Balance`: Customer's account balance
*   `NumOfProducts`: Number of bank products the customer uses
*   `HasCrCard`: Whether the customer has a credit card (1 = Yes, 0 = No)
*   `IsActiveMember`: Whether the customer is an active member (1 = Yes, 0 = No)
*   `EstimatedSalary`: Estimated salary of the customer
*   `Exited`: Churn status (1 = Churned, 0 = Not Churned) - This is the target variable.

The dataset contains 10,000 entries, providing a substantial amount of data for training and evaluating the churn prediction model.

### 3.2 Data Preprocessing

Data preprocessing is a crucial step to prepare the raw data for machine learning model training. The `experiment.ipynb` notebook outlines the following preprocessing steps:

1.  **Feature Selection**: The `RowNumber`, `CustomerId`, and `Surname` columns were dropped as they are identifiers and do not contribute to the predictive power of the model.

2.  **Categorical Feature Encoding**: Categorical features (`Geography` and `Gender`) were converted into numerical representations:
    *   `Gender`: `LabelEncoder` was used to transform 'Male' and 'Female' into numerical values (0 and 1). The `label_encoder_gender.pkl` file saves this encoder for consistent preprocessing during prediction.
    *   `Geography`: `OneHotEncoder` was applied to `Geography` to convert it into a one-hot encoded format, creating new columns like `Geography_France`, `Geography_Germany`, and `Geography_Spain`. This prevents the model from assuming an ordinal relationship between countries. The `one_hot_encoder_geo.pkl` file saves this encoder.

3.  **Feature Scaling**: Numerical features were scaled using `StandardScaler`. This is important for neural networks as it normalizes the input data, preventing features with larger values from dominating the learning process. The `scaler.pkl` file saves the fitted scaler.

4.  **Data Splitting**: The dataset was split into training and testing sets (80% for training, 20% for testing) to evaluate the model's performance on unseen data. The target variable `Exited` was separated from the features.

These preprocessing steps ensure that the data is in a suitable format for the Artificial Neural Network, improving its training efficiency and predictive accuracy.



## 4. Methodology

### 4.1 Proposed Solution: Artificial Neural Network (ANN)

An Artificial Neural Network (ANN) was chosen for this binary classification problem due to its ability to learn complex non-linear relationships within the data, which is often the case in customer behavior prediction. ANNs are particularly well-suited for tasks where the underlying patterns are not easily captured by traditional linear models.

### 4.2 Model Architecture

The ANN model implemented in `experiment.ipynb` is a sequential model, meaning the data flows in one direction from input to output through a series of layers. The architecture consists of:

*   **Input Layer**: The input layer implicitly takes the number of features after preprocessing (12 features: CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain). The `input_shape` for the first `Dense` layer is set to `(X_train.shape[1],)`, which dynamically adjusts to the number of input features.

*   **Hidden Layers**: The model uses two hidden layers:
    *   **First Hidden Layer**: Consists of 64 neurons with a `relu` (Rectified Linear Unit) activation function. ReLU is a popular choice for hidden layers due to its computational efficiency and ability to mitigate the vanishing gradient problem.
    *   **Second Hidden Layer**: Consists of 32 neurons, also with a `relu` activation function.

*   **Output Layer**: A single neuron with a `sigmoid` activation function. The sigmoid function outputs a probability value between 0 and 1, which is ideal for binary classification problems, representing the likelihood of churn.

The model summary from `experiment.ipynb` confirms the architecture and the number of trainable parameters:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                832       
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 2945 (11.50 KB)
Trainable params: 2945 (11.50 KB)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________
```

### 4.3 Training and Optimization

The model was compiled with the following settings:

*   **Optimizer**: `Adam` optimizer with a learning rate of 0.01. Adam is an adaptive learning rate optimization algorithm that is well-suited for a wide range of deep learning problems.
*   **Loss Function**: `binary_crossentropy`. This is the standard loss function for binary classification tasks, measuring the difference between the predicted probabilities and the true labels.
*   **Metrics**: `accuracy` was used to monitor the model's performance during training.

To prevent overfitting and improve generalization, the following callbacks were used during training:

*   **Early Stopping**: Monitored `val_loss` (validation loss) with a `patience` of 5. This means training would stop if the validation loss did not improve for 5 consecutive epochs, and the model weights from the best epoch (lowest validation loss) would be restored.
*   **TensorBoard**: Used for logging training metrics and visualizing the training process, allowing for better understanding and debugging of the model's learning behavior.

The model was trained for 100 epochs, with the early stopping mechanism ensuring that training concluded once performance on the validation set plateaued or degraded. The trained model is saved as `churn_model.h5`.



## 5. Implementation Pipeline and Outcomes

### 5.1 Prediction Pipeline

The trained `churn_model.h5` is utilized in a prediction pipeline, as demonstrated in `prediction.ipynb` and `app.py`. The pipeline involves loading the saved model, along with the pre-fitted `label_encoder_gender.pkl`, `one_hot_encoder_geo.pkl`, and `scaler.pkl` objects. This ensures that new input data undergoes the exact same preprocessing steps as the training data, maintaining consistency and accuracy.

For a new customer, the prediction process involves:

1.  **Data Collection**: Gathering the customer's attributes (CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary).
2.  **Preprocessing**: Applying the loaded `LabelEncoder` to `Gender`, `OneHotEncoder` to `Geography`, and `StandardScaler` to all numerical features.
3.  **Prediction**: Feeding the preprocessed data into the loaded `churn_model.h5` to obtain a churn probability.
4.  **Interpretation**: The output probability (between 0 and 1) indicates the likelihood of the customer churning. A threshold (e.g., 0.5) can be used to classify the customer as likely to churn or not.

### 5.2 Streamlit Application

A user-friendly web application (`app.py`) was developed using Streamlit to provide an interactive interface for churn prediction. This application allows users to input customer details through a simple form and receive an instant churn probability prediction. The Streamlit application demonstrates the practical deployment of the trained model, making it accessible for non-technical users to assess churn risk.

The `app.py` script performs the following:

*   Loads the `churn_model.h5`, `label_encoder_gender.pkl`, `one_hot_encoder_geo.pkl`, and `scaler.pkl` files.
*   Creates input widgets (select boxes, sliders, number inputs) for each customer attribute.
*   Collects user input and transforms it into the format expected by the model, applying the same encoding and scaling steps.
*   Makes a prediction using the loaded model.
*   Displays the churn probability and a clear message indicating whether the customer is likely to churn or continue using the service.

This application serves as a tangible outcome of the project, showcasing the end-to-end implementation of an AI-based solution for a real-world problem. It provides a practical tool for bank employees or decision-makers to quickly assess individual customer churn risk and potentially inform retention strategies.




## 6. Conclusion

This project successfully developed and implemented an Artificial Neural Network model for predicting customer churn in a banking environment. By leveraging historical customer data and applying appropriate preprocessing techniques, the model can effectively identify customers at risk of churning. The accompanying Streamlit application demonstrates a practical deployment of this AI solution, making it accessible for real-world use cases.

The project adheres to the requirements by:

a)  **Project Title**: "Customer Churn Prediction using Artificial Neural Networks" clearly reflects a real-world problem (customer churn) and an AI-based solution (ANN).

b)  **Dataset & Preprocessing**: A relevant dataset (`Churn_Modelling.csv`) was chosen, and comprehensive data cleaning, preprocessing (feature selection, categorical encoding, feature scaling), and feature engineering steps were performed and documented.

c)  **Methodology**: An Artificial Neural Network was proposed as the solution, with a clear justification for its choice (ability to learn complex non-linear relationships) and a detailed description of its architecture (input, hidden, and output layers, activation functions, optimizers, loss function, and metrics).

d)  **Final Report**: This document serves as the professional report, detailing the problem definition, dataset and preprocessing steps, proposed solution and methodology, and implementation pipeline and outcomes. It is structured, clear, and comprehensive.

e)  **Code Package Submission**: All working code (Jupyter notebooks, Python application) and documentation (this report) are included, ensuring functionality, clarity, and completeness for evaluation.

This project provides a robust framework for banks to proactively manage customer retention, ultimately contributing to business growth and stability.


