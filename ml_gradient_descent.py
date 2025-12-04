import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------
# TRAINING DATA
# -------------------------------------------
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)  # y = 2x line

# -------------------------------------------
# COST FUNCTION: MSE
# -------------------------------------------
def compute_cost(x, y, w, b):
    m = len(x)
    total = 0
    for i in range(m):
        f_wb = w * x[i] + b
        total += (f_wb - y[i]) ** 2
    return total / (2 * m)


# -------------------------------------------
# GRADIENT FUNCTION
# -------------------------------------------
def compute_gradient(x, y, w, b):
    m = len(x)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        dj_dw += error * x[i]
        dj_db += error
    
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


# -------------------------------------------
# GRADIENT DESCENT
# -------------------------------------------
def gradient_descent(x, y, w, b, alpha, num_iters):
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Update rule
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)

        if i % 50 == 0:
            print(f"Iteration {i}: Cost={cost:.4f}, w={w:.4f}, b={b:.4f}")

    return w, b, cost_history


# -------------------------------------------
# MAIN EXECUTION
# -------------------------------------------
w_init = 0
b_init = 0
alpha = 0.01
iterations = 500

w_final, b_final, cost_history = gradient_descent(x, y, w_init, b_init, alpha, iterations)

print("\nTraining completed!")
print("Final w:", w_final)
print("Final b:", b_final)

# -------------------------------------------
# PLOT COST OVER ITERATIONS
# -------------------------------------------
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent Cost Over Time")
plt.show()

# -------------------------------------------
# PLOT COST OVER ITERATIONS
# -------------------------------------------
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent Cost Over Time")
plt.show()

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(x, w, b):
    return w * x + b

print("\nTraining complete!")
print(f"Final w = {w_final}")
print(f"Final b = {b_final}")

# Predict house prices for new input sizes
x_test = np.array([1200, 1500, 2000, 3000])

for x_val in x_test:
    y_pred = predict(x_val, w_final, b_final)
    print(f"Predicted price for house {x_val} sq-ft = {y_pred:.2f}")

