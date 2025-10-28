Eigenvalues and eigenvectors are fundamental concepts in linear algebra that provide profound insights into the behavior of linear transformations. They are particularly important when dealing with square matrices.

### The Core Idea: What remains "Eigen"?

The word "eigen" comes from German and means "proper" or "characteristic." In essence, eigenvalues and eigenvectors reveal the **characteristic directions** along which a linear transformation acts merely by stretching or shrinking vectors, without changing their direction (or only reversing it).

>Imagine a linear transformation, like rotating, stretching, or shearing a shape in space. Most vectors will change both their direction and magnitude. However, there are special vectors that, when acted upon by the transformation, only get scaled (stretched or shrunk), while their direction remains the same (or perfectly opposite). These special vectors are the **eigenvectors**, and the factor by which they are scaled is the **eigenvalue**.

### Defining the Relationship: The Eigenvalue Equation

Mathematically, this relationship is expressed by the **eigenvalue equation**:

Av=λv

Where:

- A is a square matrix representing the linear transformation.
- v is a non-zero vector, which is the **eigenvector**.
- λ (lambda) is a scalar value, which is the **eigenvalue** corresponding to the eigenvector v.

Let's break this down:

- **When A multiplies v (Av)**: This represents applying the linear transformation defined by matrix A to the vector v.
- **When v is multiplied by λ (λv)**: This means the vector v is simply scaled by a factor of λ.

The equation Av=λv states that when the transformation A is applied to the eigenvector v, the result is simply a scaled version of v. The direction of v does not change (unless λ is negative, which just reverses the direction).

### Key Characteristics:

- **Eigenvectors are non-zero**: If v were the zero vector, the equation A⋅0=λ⋅0 would always be true for any λ, making the concept meaningless.
- **Eigenvalues can be zero, positive, or negative, or even complex**:
    - If λ=1, the eigenvector's length and direction remain unchanged.
    - If λ>1, the eigenvector is stretched.
    - If 0<λ<1, the eigenvector is shrunk.
    - If λ<0, the eigenvector's direction is reversed and scaled.
    - If λ=0, the eigenvector is mapped to the zero vector (meaning it's in the null space of A).
- **An eigenvector defines a direction, not a specific vector**: If v is an eigenvector with eigenvalue λ, then any non-zero scalar multiple of v (e.g., 2v, −5v) is also an eigenvector with the same eigenvalue. This is because A(cv)=c(Av)=c(λv)=λ(cv).
- **Eigenspace**: For a given eigenvalue λ, all the eigenvectors corresponding to that λ (along with the zero vector) form a subspace called the **eigenspace**.

### Why are they important? (Applications)

Eigenvalues and eigenvectors are incredibly powerful tools in various fields of science, engineering, and data analysis because they help us understand the fundamental properties and behaviors of systems described by linear transformations. Here are some examples:

- **Principal Component Analysis (PCA)**: In data science, PCA uses eigenvalues and eigenvectors to reduce the dimensionality of data while retaining as much variance as possible. Eigenvectors represent the "principal components" (directions of greatest variance), and eigenvalues indicate the amount of variance along those directions.
- **Google's PageRank Algorithm**: The original PageRank algorithm, which determines the importance of web pages, is based on finding the principal eigenvector of a matrix representing the link structure of the web.
- **Vibration Analysis**: In mechanical engineering, eigenvalues represent the natural frequencies of vibration of a structure, and eigenvectors represent the corresponding mode shapes (how the structure deforms during vibration). This is crucial for designing buildings, bridges, and other structures that can withstand resonance.
- **Quantum Mechanics**: In quantum mechanics, eigenvalues represent the possible measurable values of physical quantities (like energy or momentum), and eigenvectors represent the corresponding quantum states.
- **Stability Analysis**: Eigenvalues can be used to determine the stability of dynamical systems, whether it's a control system, an ecosystem, or a chemical reaction.
- **Image Compression**: Techniques like Singular Value Decomposition (SVD), which relies on eigenvalues, are used to compress images by identifying and keeping the most significant information.
- **Facial Recognition**: "Eigenfaces" are a classic application where eigenvectors of image datasets are used to represent and recognize faces efficiently.

In essence, eigenvalues and eigenvectors help us find the "natural" or "characteristic" behaviors within complex linear systems, simplifying their analysis and providing profound insights.

### How to Compute
Computing eigenvalues and eigenvectors involves a two-step process:

1.  **Find the eigenvalues (λ):** These are the scalar values.
2.  **Find the eigenvectors (v) for each eigenvalue:** These are the non-zero vectors.

Let's break down the steps using the fundamental eigenvalue equation: $Av = \lambda v$

### Step 1: Finding the Eigenvalues (λ)

The core idea is to transform the eigenvalue equation into a form that allows us to solve for $\lambda$.

Starting with $Av = \lambda v$:

1.  Move $\lambda v$ to the left side:
    $Av - \lambda v = 0$

2.  Factor out $v$. To do this, we need to introduce the identity matrix $I$ (of the same dimension as $A$) so that $\lambda v$ can be written as $\lambda I v$:
    $Av - \lambda I v = 0$
    $(A - \lambda I) v = 0$

Now, we have a homogeneous system of linear equations. For this system to have **non-trivial solutions** (i.e., solutions where $v \neq 0$, which is a requirement for eigenvectors), the matrix $(A - \lambda I)$ **must be singular**. A singular matrix has a determinant of zero.

Therefore, to find the eigenvalues, we solve the **characteristic equation**:

$\det(A - \lambda I) = 0$

**How to compute $\det(A - \lambda I)$:**

* **For a 2x2 matrix $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:**
    $A - \lambda I = \begin{pmatrix} a & b \\ c & d \end{pmatrix} - \lambda \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} a-\lambda & b \\ c & d-\lambda \end{pmatrix}$
    $\det(A - \lambda I) = (a-\lambda)(d-\lambda) - bc = 0$
    This will result in a quadratic equation in $\lambda$ (a characteristic polynomial of degree 2). Solve this quadratic equation to find the two eigenvalues.

* **For a 3x3 matrix $A = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$:**
    $A - \lambda I = \begin{pmatrix} a_{11}-\lambda & a_{12} & a_{13} \\ a_{21} & a_{22}-\lambda & a_{23} \\ a_{31} & a_{32} & a_{33}-\lambda \end{pmatrix}$
    Computing the determinant of a 3x3 matrix (using cofactor expansion or Sarrus' rule) will result in a cubic equation in $\lambda$ (a characteristic polynomial of degree 3). Solve this cubic equation to find the three eigenvalues.

**General Case (n x n matrix):**
For an $n \times n$ matrix, computing $\det(A - \lambda I)$ will yield a characteristic polynomial of degree $n$. The roots of this polynomial are the eigenvalues. Finding the roots of higher-degree polynomials can be challenging by hand, and numerical methods are often used for matrices larger than 3x3.

### Step 2: Finding the Eigenvectors (v)

Once you have found the eigenvalues ($\lambda_1, \lambda_2, ..., \lambda_n$), you need to find the corresponding eigenvectors for each one.

For each eigenvalue $\lambda_i$:

1.  Substitute $\lambda_i$ back into the equation $(A - \lambda I) v = 0$:
    $(A - \lambda_i I) v = 0$

2.  Solve this homogeneous system of linear equations for $v$. This means finding the null space (or kernel) of the matrix $(A - \lambda_i I)$.

    You can use techniques like **Gaussian elimination** (row reduction) to find the general solution for $v$. The solution will be a set of vectors that span the eigenspace corresponding to $\lambda_i$. Any non-zero vector in this eigenspace is an eigenvector for $\lambda_i$.

**Example for a 2x2 matrix:**

Let's say you have a matrix $A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}$.

**Step 1: Find Eigenvalues**

1.  Form $(A - \lambda I)$:
    $A - \lambda I = \begin{pmatrix} 4-\lambda & 2 \\ 1 & 3-\lambda \end{pmatrix}$

2.  Compute the determinant and set it to zero:
    $\det(A - \lambda I) = (4-\lambda)(3-\lambda) - (2)(1) = 0$
    $12 - 4\lambda - 3\lambda + \lambda^2 - 2 = 0$
    $\lambda^2 - 7\lambda + 10 = 0$

3.  Solve the quadratic equation:
    $(\lambda - 2)(\lambda - 5) = 0$
    So, the eigenvalues are $\lambda_1 = 2$ and $\lambda_2 = 5$.

**Step 2: Find Eigenvectors**

**For $\lambda_1 = 2$:**

1.  Substitute $\lambda = 2$ into $(A - \lambda I) v = 0$:
    $(A - 2I) v = \begin{pmatrix} 4-2 & 2 \\ 1 & 3-2 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$
    $\begin{pmatrix} 2 & 2 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

2.  Solve the system:
    $2x + 2y = 0 \implies x + y = 0 \implies x = -y$
    $1x + 1y = 0 \implies x + y = 0 \implies x = -y$
    Let $y = t$ (where $t$ is a free variable and $t \neq 0$). Then $x = -t$.
    So, the eigenvectors for $\lambda_1 = 2$ are of the form $\begin{pmatrix} -t \\ t \end{pmatrix} = t \begin{pmatrix} -1 \\ 1 \end{pmatrix}$.
    A common choice for the eigenvector is $v_1 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$ (by setting $t=1$).

**For $\lambda_2 = 5$:**

1.  Substitute $\lambda = 5$ into $(A - \lambda I) v = 0$:
    $(A - 5I) v = \begin{pmatrix} 4-5 & 2 \\ 1 & 3-5 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$
    $\begin{pmatrix} -1 & 2 \\ 1 & -2 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

2.  Solve the system:
    $-x + 2y = 0 \implies x = 2y$
    $x - 2y = 0 \implies x = 2y$
    Let $y = s$ (where $s$ is a free variable and $s \neq 0$). Then $x = 2s$.
    So, the eigenvectors for $\lambda_2 = 5$ are of the form $\begin{pmatrix} 2s \\ s \end{pmatrix} = s \begin{pmatrix} 2 \\ 1 \end{pmatrix}$.
    A common choice for the eigenvector is $v_2 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$ (by setting $s=1$).

This process allows you to systematically compute the eigenvalues and their corresponding eigenvectors for any square matrix. Keep in mind that for larger matrices, computational tools (like MATLAB, NumPy in Python, or specialized calculators) are often used due to the complexity of solving higher-degree polynomials and large systems of equations.


- The eigenvectors corresponding to different eigenvalues are linearly independent.
- The eigenvectors of a square symmetric matrix are orthogonal.
- The eigenvectors of a square symmetric matrix can thus form a convenient basis.

----

