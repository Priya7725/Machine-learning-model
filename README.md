# Machine-learning-model

Sure thing, Priya! Implementing a machine learning model using scikit-learn is a fascinating process. Here’s a quick guide to get you started:

1. **Install scikit-learn**: Make sure you have scikit-learn installed. You can install it using pip:
    ```bash
    pip install scikit-learn
    ```

2. **Import Libraries**: First, import the necessary libraries.
    ```python
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    ```

3. **Load Dataset**: Load your dataset. For example, let’s use the famous Iris dataset:
    ```python
    from sklearn.datasets import load_iris
    data = load_iris()
    X = data.data
    y = data.target
    ```

4. **Split Dataset**: Split the data into training and testing sets.
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

5. **Choose and Train a Model**: Choose a model (e.g., Random Forest) and train it.
    ```python
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    ```

6. **Make Predictions**: Use the trained model to make predictions on the test set.
    ```python
    y_pred = model.predict(X_test)
    ```

7. **Evaluate the Model**: Evaluate the model’s performance.
    ```python
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    ```
