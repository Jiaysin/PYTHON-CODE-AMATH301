import numpy as np


np.random.seed(1)


Ajacobi = 0.01 * np.random.randint(-50, 51, [10, 10]) + 10 * np.eye(10)
bjacobi = np.random.randint(-50, 51, [10, 1])


x0 = np.zeros(10)

tolerance = 1e-5
max_iterations = 1000


stepvec = []


xn = x0.copy()

for j in range(max_iterations):
    xn1 = np.zeros_like(xn)  # Create a new x_n+1

    # Update each element of x using the Jacobi formula
    for i in range(10):
        # Compute the sum of A[i, j] * x[j] for j != i
        sum_Ax = 0
        for j_col in range(10):
            if j_col != i:
                sum_Ax += Ajacobi[i, j_col] * xn[j_col]

        # Update x_n+1[i]
        xn1[i] = (bjacobi[i] - sum_Ax) / Ajacobi[i, i]

    # Compute step norm (2-norm of the difference between xn1 and xn)
    step = np.linalg.norm(xn1 - xn, 2)
    stepvec.append(step)

    # Check for convergence
    if step < tolerance:
        print(f"Converged at step {j} with step size {step}")
        break

    # Update xn for the next iteration
    xn = xn1.copy()


if len(stepvec) > 7:
    stepvec = stepvec[:7]  # Truncate to 7 values
elif len(stepvec) < 7:
    stepvec = np.append(stepvec, [0] * (7 - len(stepvec)))  # Pad with zeros if necessary


stepvec = np.array(stepvec)


print("Final solution x:", xn)
print("Step vector:", stepvec)