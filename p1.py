import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

df_independent = pd.read_csv("logisticX.csv") 
df_dependent = pd.read_csv("logisticY.csv")

y = df_dependent.iloc[:, 0]
X = df_independent

X_squared = pd.concat([X, X**2], axis=1)

def train_logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    theta = np.zeros(X.shape[1])  # Initialize theta vector with zeros
    cost_history = []

    for i in range(iterations):
        # Calculate hypothesis (predicted probabilities)
        hypothesis = 1 / (1 + np.exp(-np.dot(X, theta)))

        gradient = np.dot(X.T, (hypothesis - y))

        theta -= learning_rate * gradient

        cost = np.mean(-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis))
        cost_history.append(cost)

    return theta, cost_history

theta_original, cost_history_original = train_logistic_regression(X, y)
theta_squared, cost_history_squared = train_logistic_regression(X_squared, y)

def plot_decision_boundary(X, y, theta, title="Logistic Regression Decision Boundary"):
    plt.figure() 
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors)

    x1_min, x1_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    x2_min, x2_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    z = 1 / (1 + np.exp(-np.dot(np.c_[xx.ravel(), yy.ravel()], theta)))
    z = z.reshape(xx.shape)
    plt.contour(xx, yy, z, 1, colors='black')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.show()

plot_decision_boundary(X, y, theta_original, title="Decision Boundary (Original Features)")
plot_decision_boundary(X_squared, y, theta_squared, title="Decision Boundary (Squared Features)")
plt.figure()
plt.plot(range(len(cost_history_original[:50])), cost_history_original[:50])  # Plot first 50 iterations
plt.xlabel("Iteration")
plt.ylabel("Cost Function")
plt.title("Cost Function vs. Iteration (Original Features)")
plt.show()

model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)

confusion_matrix_result = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", confusion_matrix_result)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("Final Cost:", cost_history_original[-1])
print("Final Coefficients (Original Features):", theta_original)
