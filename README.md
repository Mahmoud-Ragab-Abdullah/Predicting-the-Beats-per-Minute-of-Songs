# Predicting-the-Beats-per-Minute-of-Songs
An analysis of songs dataset to predict the BeatsPerMinute using machine learning models

Project Overview
This project is a data science and machine learning analysis of a song dataset with the goal of predicting a song's BeatsPerMinute (BPM). The BPM, a direct measure of a song's tempo, is an important feature in music analysis. The target class for this project is a numeric feature, which makes this a regression problem.

Data Source
The dataset used in this project is sourced from a Kaggle competition. The dataset contains various audio features that serve as predictors for the target variable BeatsPerMinute. Key features in the dataset include:

RhythmScore: A measure of rhythmic complexity or regularity.

AudioLoudness: The average loudness of the audio track.

VocalContent: The proportion of the track dominated by vocals.

InstrumentalScore: A measure of how much of the track is purely instrumental.

Energy: A perceptual measure of intensity and activity.

TrackDurationMs: The duration of the song in milliseconds.

Methodology
My approach to this project followed a standard data science workflow, from data exploration to model training and evaluation.

Data Loading and Inspection: The training and testing datasets were loaded using the pandas library. An initial inspection revealed that there were no missing values in the dataset, and all predictor features were numeric.

Exploratory Data Analysis (EDA): I performed a detailed EDA to understand the dataset's characteristics. Visualizations were used to plot the relationships between BeatsPerMinute and other features. The plots did not show a strong, obvious linear relationship.

Preprocessing: I determined that the data needed scaling before training because each feature had a different scale. A StandardScaler was included in the final model pipeline to handle this.

Model Building and Training: Based on the non-linear relationships observed during the EDA, I chose to use a RandomForestRegressor to train the model. I used a StandardScaler within a make_pipeline to ensure proper data scaling before training.

Model Evaluation: The model was evaluated using a learning curve, plotting the Root Mean Squared Error (RMSE).


Based on the provided learning curve graph, here is an altered and more detailed "Results & Conclusion" section for your README.

Results & Conclusion
The learning curve plot provides a clear evaluation of the model's performance as more data is added.

Training Score (Red Line): The training error is low at the beginning with a small number of training examples, indicating the model is fitting the training data very well. As the number of training examples increases, the training error steadily rises, which is a normal behavior as the model has a harder time fitting a larger, more complex dataset.

Cross-Validation Score (Green Line): The cross-validation error, which represents the model's performance on unseen data, remains consistently high. It shows very little improvement even as more data is added.

Use a more complex model: Consider a more powerful regressor like Gradient Boosting or an XGBoost.

Tune hyperparameters: Increase the complexity of the current Random Forest Regressor by tuning parameters such as max_depth or n_estimators.

Perform more feature engineering: Explore creating new features from the existing ones to give the model more information.

Technologies Used
Programming Language: Python

Libraries: pandas, numpy, scikit-learn, matplotlib
