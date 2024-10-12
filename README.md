# Student Score Analysis and Prediction Using Machine Learning

This project analyzes a dataset of student scores and uses machine learning techniques to predict scores based on study hours.

## Dataset Description

The dataset contains information about students' study hours and the scores they earned. It consists of two columns:
- Hours: Number of study hours
- Scores: Scores obtained

## Analysis Steps

1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. Machine Learning Modeling
4. Model Evaluation

## Analysis Results

### Exploratory Data Analysis

- There is a strong positive correlation between study hours and scores (correlation: 0.9761)
- The distribution of study hours is skewed to the right
- The distribution of scores is relatively normal

### Feature Engineering

- No duplicate data found
- There are no missing values
- Some potential outliers are detected, but the number is not significant

### Modeling and Evaluation

Three regression models were used:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Model evaluation results:

| Model MSE | RMSE | MAE | R2 Score |
|--------------------|---------|---------|---------|----------|
| Linear Regression | 21.5987 | 4.6474 | 3.9419 | 0.9477 |
| Decision Tree | 25.4467 | 5.0445 | 3.7333 | 0.9384 |
| Random Forest | 17.9537 | 4.2372 | 3.1667 | 0.9566 |

The Random Forest model shows the best performance with the highest R2 Score (0.9566) and the lowest MSE (17.9537).

## Conclusion

Based on R2 Score, the Random Forest model has the best performance. However, the final model selection should consider the trade-off between performance, complexity, and interpretability according to the specific needs of the project.

## How to Run the Code

This project was run using Google Colab. To run the code:
1. Open the notebook file in Google Colab
2. Upload the dataset 'student_scores.csv' to the Colab environment
3. Run the code cells sequentially

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Contributions

Contributions to the improvement and development of this project are very welcome. Please create a pull request or open an issue for discussion.

## License

[MIT License](https://opensource.org/licenses/MIT)
