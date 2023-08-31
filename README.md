# Fuel-Efficiency-Prediction-Model-and-Web-Service-with-Python

vehicle fuel consumption.ipynb contains the whole pyhton code, and rest are dependencies. Below is the complete breakdown of the code. 

Here's a step-by-step breakdown of the code:

Importing Libraries: The code starts by importing necessary libraries for data processing, visualization, and machine learning model selection. These include NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn's various modules.

Reading Data: The code reads data from a CSV file named "auto-mpg.data" into a Pandas DataFrame. The data contains information about vehicles, including their features and MPG values.

Data Splitting: The data is split into training and testing sets using StratifiedShuffleSplit. This ensures that the distribution of a specific feature, "Cylinders," remains similar in both sets.

Data Preprocessing: The "Origin" column in the dataset is preprocessed using a function preprocess_origin_cols() to map numerical values to categorical values like "India," "USA," and "Germany." This makes it more understandable for the machine learning model.

Custom Attribute Adder: A custom attribute adder class CustomAttrAdder is defined, which calculates additional attributes based on the existing features. For example, it calculates acceleration per horsepower and acceleration per cylinder. This class will be used later in the data transformation pipeline.

Numerical Pipeline: A function num_pipeline_transformer() is defined to create a pipeline for numerical feature transformations. This pipeline includes imputing missing values, adding custom attributes, and standardizing the features using StandardScaler.

Full Transformation Pipeline: A function pipeline_transformer() is defined to create the full transformation pipeline. This pipeline handles both numerical and categorical features. Numerical features are processed using the previously defined numerical pipeline, and categorical features are one-hot encoded using OneHotEncoder.

Preparing Data: The raw training data is preprocessed using the origin preprocessing function, and then the full transformation pipeline is applied to create the prepared_data.

Model Selection and Training: Linear Regression, Decision Tree Regression, Random Forest Regression, and Support Vector Regression models are selected for training and comparison. The models are trained using the prepared data and corresponding labels.

Cross-Validation Testing: Cross-validation is performed to evaluate the models' performance using negative mean squared error as the evaluation metric. This provides an estimate of how well the models generalize to new data.

Hyperparameter Tuning: GridSearchCV is used to perform hyperparameter tuning for the Random Forest Regression model. Different combinations of hyperparameters are tested, and the best combination is selected based on cross-validation results.

Feature Importance: The importance of each feature in the best Random Forest model is calculated and displayed, showing which features contribute the most to the predictions.

Saving and Loading Models: The best model (Random Forest) is saved to a binary file using the pickle library. It is then loaded back from the file to demonstrate the loading process.

Web Service Development: The code demonstrates how to develop a simple web service for making predictions using the trained model. It first installs the httpx library and then provides an example of making a POST request to the locally hosted service, sending vehicle data for prediction. However, there seems to be an issue with this part of the code, as it's raising an error related to JSON parsing. 

To package a machine learning model into a web service, you can use the Flask web framework. Flask is a lightweight framework that makes it easy to develop web services.

To get started, you need to create a new directory for your flask application. Then, you need to install the Flask and other necessary packages.

Once you have installed the necessary packages, you can create a new file called main.py. In this file, you need to create a Flask app and define a route that will accept the vehicle_config from an HTTP POST request and return the predictions using the model and predict_mpg() method.

You also need to load the trained model into the model variable from the file you have in the model_files folder.

Finally, you need to start the server running the main.py file.

You can test this route using Postman or the requests package.

Here is an example of how you can test the route using Postman:

Open Postman.
Create a new request.
Set the request method to POST.
Set the request URL to http://localhost:9696/predict.
In the body of the request, add the vehicle_config as a JSON object.
Click Send.
The response from the server will be a JSON object containing the mpg_prediction.

I hope this helps! Let me know if you have any other questions. 

Follow me on medium, to view the detailed explanation of this code: https://rajdeepsarkar95.medium.com/
