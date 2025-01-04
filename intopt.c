/* 
 * intopt.c
 * 
 * Implementation of the Simplex algorithm and Integer Linear Programming (ILP) using Branch-and-Bound.
 * 
 * Author: Ali Bakly
 * Date: 2024-12-07
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>


/* ------------------------------------------------------------
 * Preprocessor Directives and Constants
 * ------------------------------------------------------------ */


/**
 * @brief A small threshold value to handle floating-point precision.
 */
#define EPSILON 1e-6


/* ------------------------------------------------------------
 * Type Definitions
 * ------------------------------------------------------------ */


/**
 * @brief Structure representing the Simplex tableau.
 */
typedef struct simplex_t {
    int m;               /**< Number of constraints */
    int n;               /**< Number of decision variables */
    int* var;            /**< Array of size (n + m + 1) for variable indices */
    double** a;          /**< 2D array of size m x (n + 1) representing the tableau */
    double* b;           /**< Array of size m for right-hand side values */
    double* x;           /**< Array of size (n + 1) for solution variables */
    double* c;           /**< Array of size n for objective function coefficients */
    double y;            /**< Objective function value */
} simplex_t;

/**
 * @brief Structure representing a node in the Branch-and-Bound tree.
 */
typedef struct node_t {
    int m;          /**< Number of constraints */
    int n;          /**< Number of decision variables */
    int k;          /**< Parent branches on xk */
    int h;          /**< Branch on xh */
    double xh;      /**< xh value */
    double ak;      /**< Parent ak */
    double bk;      /**< Parent bk */
    double* min;    /**< Lower bounds */
    double* max;    /**< Upper bounds */
    double** a;     /**< Constraint matrix */
    double* b;      /**< Right-hand side values */
    double* x;      /**< Solution variables */
    double* c;      /**< Objective function coefficients */
    double z;       /**< Objective function value */
} node_t;

/**
 * @brief Structure representing a node in the stack.
 */
typedef struct linked_node_stack_t {
    node_t* node;                    /**< Pointer to the data node */
    struct linked_node_stack_t* next; /**< Pointer to the next list node */
} linked_node_stack_t;


/* ------------------------------------------------------------
 * Function Prototypes
 * ------------------------------------------------------------ */


int init(simplex_t* s, int m, int n, double** a, double* b, double* c, double* x, double y, int* var);
int initial(simplex_t* s, int m, int n, double** a, double* b, double* c, double* x, double y, int* var);
int select_nonbasic(simplex_t* s);
void pivot(simplex_t* s, int row, int col);
void prepare(simplex_t* s, int k);
double xsimplex(int m, int n, double** a, double* b, double* c, double* x, double y, int* var, int h);
double simplex(int m, int n, double** a, double* b, double* c, double* x, double y);

/* ILP-related functions */
node_t* initial_node(int m, int n, double** a, double* b, double* c);
node_t* extend(node_t* p, int m, int n, double** a, double* b, double* c, int k, double ak, double bk);
int is_integer(double* xp);
int integer(node_t* p);
node_t* pop(linked_node_stack_t** h);
void push(linked_node_stack_t** h, node_t* node);
void free_node(node_t* q);
void bound(node_t* p, linked_node_stack_t** h, double* zp, double* x);
int branch(node_t* q, double z);
void succ(node_t* p, linked_node_stack_t** h, int m, int n, double** a, double* b, double* c, int k, double ak, double bk, double* zp, double* x);
double intopt(int m, int n, double** a, double* b, double* c, double* x);


/* ------------------------------------------------------------
 * Helper Functions
 * ------------------------------------------------------------ */


/* ------------------------------------------------------------
 * Function Implementations
 * ------------------------------------------------------------ */


/* 
 * Simplex-related functions
 */

/**
 * @brief Initializes the simplex structure with given parameters.
 *
 * @param s Pointer to the simplex_t structure to initialize.
 * @param m Number of constraints.
 * @param n Number of decision variables.
 * @param a 2D array representing the constraint matrix.
 * @param b Array representing the right-hand side values.
 * @param c Array representing the objective function coefficients.
 * @param x Array to store the solution variables.
 * @param y Objective function value.
 * @param var Array for variable indices.
 *
 * @return The index of the constraint with the smallest b[k].
 */
int init(simplex_t* s, int m, int n, double** a, double* b, double* c, double* x, double y, int* var) {
    int i;
    int k;

    // Assign values to the simplex structure
    s->m = m;
    s->n = n;
    s->a = a;
    s->b = b;
    s->c = c;
    s->x = x;
    s->y = y;
    s->var = var; 

    // Initialize variable indices if not provided
    if (s->var == NULL) {
        s->var = malloc((s->m + s->n + 1) * sizeof(int));
        for (i = 0; i < m + n; i++) {
            s->var[i] = i;
        }
    }

    // Find the index k with the smallest b[k]
    for (k = 0, i = 1; i < m; i++) {
        if (b[i] < b[k]) {
            k = i;
        }
    }

    return k;
}

/**
 * @brief Checks the initial feasibility of the simplex tableau and prepares it for the simplex algorithm.
 *
 * This function initializes the simplex tableau, determines if the current solution is feasible,
 * and prepares the tableau by introducing slack variables if necessary. It also updates the
 * objective function coefficients based on the current tableau.
 *
 * @param s   Pointer to the simplex_t structure representing the simplex tableau.
 * @param m   Number of constraints in the linear program.
 * @param n   Number of decision variables in the linear program.
 * @param a   2D array representing the constraint matrix (size m x n).
 * @param b   Array representing the right-hand side values of the constraints (size m).
 * @param c   Array representing the coefficients of the objective function (size n).
 * @param x   Array to store the solution variables (size m + n).
 * @param y   Current value of the objective function.
 * @param var Array representing the current basic and non-basic variables (size m + n + 1).
 *
 * @return Returns 1 if the initial solution is feasible, otherwise 0 indicating infeasibility.
 */
int initial(simplex_t* s, int m, int n, double** a, double* b, double* c, double* x, double y, int* var) {
    int i, j, k;
    double w;

    // Initialize the simplex structure and find the constraint with the smallest b[k]
    k = init(s, m, n, a, b, c, x, y, var);
    if (b[k] >= 0) {
        return 1; // Feasible
    }

    // Prepare the simplex tableau by adding slack variables
    prepare(s, k);
    n = s->n;

    // Perform the initial simplex phase
    s->y = xsimplex(m, n, s->a, s->b, s->c, s->x, 0, s->var, 1);

    // Check the feasibility of the current solution
    for (i = 0; i < m + n; i++) {
        if (s->var[i] == m + n - 1) {
            if (fabs(s->x[i]) > EPSILON) {
                free(s->x);
                free(s->c);
                return 0; // Infeasible
            } else {
                break; // Variable found for further processing
            }
        }
    }

    // If the specific variable is basic, perform pivot to find a non-basic variable
    if (i >= n) {
        for (j = k = 0; k < n; k++) {
            if (fabs(s->a[i - n][k]) > fabs(s->a[i - n][j])) {
                j = k;
            }
        }
        pivot(s, i - n, j);
        i = j;
    }

    // Swap columns if necessary to maintain tableau structure
    if (i < n - 1) {
        // Swap columns i and n-1
        k = s->var[i];
        s->var[i] = s->var[n - 1];
        s->var[n - 1] = k;
        for (k = 0; k < m; k++) {
            w = s->a[k][n - 1];
            s->a[k][n - 1] = s->a[k][i];
            s->a[k][i] = w;
        }
    } else {
        // No action needed if the variable is already in the correct position
    }

    // Update the variable indices by shifting
    free(s->c);
    s->c = c;
    s->y = y;
    for (k = n - 1; k < n + m - 1; k++) {
        s->var[k] = s->var[k + 1];
    }

    // Decrement the number of decision variables
    s->n = s->n - 1;
    n = s->n;

    // Allocate temporary array to update objective function coefficients
    double* t = calloc(n, sizeof(double));

    // Update objective function coefficients based on current basic and non-basic variables
    for (k = 0; k < n; k++) {
        for (j = 0; j < n; j++) {
            if (k == s->var[j]) {
                // xk is non-basic. Add ck to the objective function coefficient
                t[j] += s->c[k];
                goto next_k; // Proceed to the next k
            }
        }
        // xk is basic. Update the objective function value and coefficients
        for (j = 0; j < m; j++) {
            if (s->var[n + j] == k) {
                break; // Found the row where xk is basic
            }
        }
        s->y += s->c[k] * s->b[j];
        for (i = 0; i < n; i++) {
            t[i] -= s->c[k] * s->a[j][i];
        }
    next_k:
        continue;
    }

    // Assign the updated coefficients back to the simplex structure
    for (i = 0; i < n; i++) {
        s->c[i] = t[i];
    }
    free(t);
    free(s->x);
    return 1;
}

/**
 * @brief Selects a non-basic variable with a positive coefficient in the objective function.
 *
 * @param s Pointer to the simplex_t structure.
 *
 * @return The index of the selected non-basic variable, or -1 if none found.
 */
int select_nonbasic(simplex_t* s) {
    int i;
    for (i = 0; i < s->n; i++) {
        if (s->c[i] > EPSILON) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Performs the pivot operation to update the simplex tableau.
 *
 * @param s Pointer to the simplex_t structure.
 * @param row The pivot row.
 * @param col The pivot column.
 */
void pivot(simplex_t* s, int row, int col) {
    double** a = s->a;
    double* b = s->b;
    double* c = s->c;
    int m = s->m;
    int n = s->n;

    // Precompute values
    double pivot_val = a[row][col];
    double inv_pivot = 1.0 / pivot_val;
    double c_col = c[col];
    double b_row = b[row];

    // Swap basic and non-basic variables
    int temp = s->var[col];
    s->var[col] = s->var[n + row];
    s->var[n + row] = temp;

    // Update objective function value
    s->y += c_col * b_row * inv_pivot;

    // Update objective function coefficients
    // c[col] = -c[col] / pivot_val
    c[col] = -c_col * inv_pivot;
    for (int i = 0; i < n; i++) {
        if (i != col) {
            c[i] -= c_col * (a[row][i] * inv_pivot);
        }
    }

    // Update right-hand side values
    // b[i] = b[i] - (a[i][col] * b[row]) / pivot_val
    for (int i = 0; i < m; i++) {
        if (i != row) {
            b[i] -= a[i][col] * b_row * inv_pivot;
        }
    }

    // Update the tableau except for the pivot row and column
    for (int i = 0; i < m; i++) {
        if (i != row) {
            double factor = a[i][col] * inv_pivot;
            for (int j = 0; j < n; j++) {
                if (j != col) {
                    a[i][j] -= factor * a[row][j];
                }
            }
        }
    }

    // Update the pivot column for non-pivot rows
    for (int i = 0; i < m; i++) {
        if (i != row) {
            a[i][col] = -a[i][col] * inv_pivot;
        }
    }

    // Update the pivot row for non-pivot columns
    for (int i = 0; i < n; i++) {
        if (i != col) {
            a[row][i] *= inv_pivot;
        }
    }

    // Normalize the pivot row
    b[row] = b_row * inv_pivot;
    a[row][col] = inv_pivot;
}


/**
 * @brief Prepares the simplex tableau by adding a slack variable and performing an initial pivot.
 *
 * This function modifies the simplex tableau to include a new slack variable, adjusts the variable
 * indices, updates the constraint matrix with the slack variable coefficients, initializes the solution
 * vector and objective function coefficients, and performs the initial pivot operation to maintain feasibility.
 *
 * @param s Pointer to the simplex_t structure representing the simplex tableau.
 * @param k Index of the constraint with the smallest b[k] value that may require introducing a slack variable.
 */
void prepare(simplex_t* s, int k) {
    int m = s->m;    // Number of constraints
    int n = s->n;    // Current number of decision variables
    int i;

    // Shift variable indices to make room for the new slack variable at position n
    for (i = m + n; i > n; i--) {
        s->var[i] = s->var[i - 1];
    }
    s->var[n] = m + n; // Assign the new slack variable index

    // Add the slack variable to each constraint with a coefficient of -1
    n = n + 1;
    for (i = 0; i < m; i++) {
        s->a[i][n - 1] = -1.0;
    }

    // Initialize the solution vector and objective function coefficients
    s->x = calloc(m + n, sizeof(double));
    s->c = calloc(n, sizeof(double));
    s->c[n - 1] = -1.0; // Coefficient for the new slack variable in the objective function

    s->n = n; // Update the number of decision variables

    // Perform the initial pivot to incorporate the slack variable into the tableau
    pivot(s, k, n - 1);
}



/**
 * @brief Executes the simplex algorithm to find the optimal solution.
 *
 * @param m Number of constraints.
 * @param n Number of decision variables.
 * @param a 2D array representing the constraint matrix.
 * @param b Array representing the right-hand side values.
 * @param c Array representing the objective function coefficients.
 * @param x Array to store the solution variables.
 * @param y Objective function value.
 * @param var Array for variable indices.
 * @param h Flag to determine the phase of the simplex method.
 *
 * @return The optimal value of the objective function, NAN if infeasible, or INFINITY if unbounded.
 */
double xsimplex(int m, int n, double** a, double* b, double* c, double* x, double y, int* var, int h) {
    simplex_t s;
    int i, row, col;
    // Initialize and check feasibility
    if (!initial(&s, m, n, a, b, c, x, y, var)) {
        free(s.var);
        return NAN; // Not a number (infeasible)
    }

    // Perform pivot operations until no entering variable is found
    while ((col = select_nonbasic(&s)) >= 0) {
        row = -1;
        for (i = 0; i < m; i++) {
            if (a[i][col] > EPSILON && 
               (row < 0 || b[i] / a[i][col] < b[row] / a[row][col])) {
                row = i;
            }
        }
        if (row < 0) {
            free(s.var);
            return INFINITY; // Unbounded
        }

        // Perform the pivot
        pivot(&s, row, col);
    }

    // Extract the solution based on the phase
    if (h == 0) {
        for (i = 0; i < n; i++) {
            if (s.var[i] < n) {
                x[s.var[i]] = 0;
            }
        }
        for (i = 0; i < m; i++) {
            if (s.var[n + i] < n) {
                x[s.var[n + i]] = s.b[i];
            }
        }
        free(s.var);
    } else {
        for (i = 0; i < n; i++) {
            x[i] = 0;
        }
        for (i = n; i < n + m; i++) {
            x[i] = s.b[i - n];
        }
    }

    return s.y;
}

/**
 * @brief Simplified interface to execute the simplex algorithm.
 *
 * @param m Number of constraints.
 * @param n Number of decision variables.
 * @param a 2D array representing the constraint matrix.
 * @param b Array representing the right-hand side values.
 * @param c Array representing the objective function coefficients.
 * @param x Array to store the solution variables.
 * @param y Objective function value.
 *
 * @return The optimal value of the objective function.
 */
double simplex(int m, int n, double** a, double* b, double* c, double* x, double y) {
    return xsimplex(m, n, a, b, c, x, y, NULL, 0);
}

/* 
 * ILP-related functions 
 */

/**
 * @brief Creates and initializes the initial node for the Branch-and-Bound tree in ILP.
 *
 * This function allocates memory for a new `node_t` structure and initializes its fields
 * based on the provided parameters. It sets up the constraint matrix, right-hand side values,
 * objective function coefficients, and variable bounds. The initial node represents the root
 * of the Branch-and-Bound tree.
 *
 * @param m Number of constraints in the ILP problem.
 * @param n Number of decision variables in the ILP problem.
 * @param a 2D array representing the constraint matrix (size m x n).
 * @param b Array representing the right-hand side values of the constraints (size m).
 * @param c Array representing the objective function coefficients (size n).
 *
 * @return A pointer to the newly created and initialized `node_t` structure.
 */
node_t* initial_node(int m, int n, double** a, double* b, double* c) {
    // Allocate memory for the node structure and initialize it to zero
    node_t* p = calloc(1, sizeof(node_t));

    // Allocate memory for the constraint matrix (m+1 x n+1)
    p->a = calloc(m + 1, sizeof(double*));
    for (int i = 0; i < m + 1; i++) {
        p->a[i] = calloc(n + 1, sizeof(double));
    }

    // Allocate memory for the right-hand side values, objective coefficients, and solution variables
    p->b = calloc(m + 1, sizeof(double));
    p->c = calloc(n + 1, sizeof(double));
    p->x = calloc(n + 1, sizeof(double));

    // Allocate memory for the variable bounds
    p->min = malloc(n * sizeof(double));
    p->max = malloc(n * sizeof(double));

    // Set the number of constraints and decision variables
    p->m = m;
    p->n = n;

    // Copy the constraint matrix from input to the node's constraint matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            p->a[i][j] = a[i][j];
        }
    }

    // Copy the right-hand side values from input to the node's RHS array
    for (int i = 0; i < m; i++) {
        p->b[i] = b[i];
    }

    // Copy the objective function coefficients from input to the node's objective array
    for (int i = 0; i < n; i++) {
        p->c[i] = c[i];
    }

    // Initialize variable bounds to represent no restrictions initially
    for (int i = 0; i < n; i++) {
        p->min[i] = -INFINITY; // No lower bound
        p->max[i] = INFINITY;  // No upper bound
    }

    return p;
}


/**
 * @brief Extends a node in the Branch-and-Bound tree for Integer Linear Programming (ILP).
 *
 * This function creates a new node (`q`) by branching on a specific variable `xk`.
 * It updates the constraints and variable bounds based on the branching direction
 * determined by the coefficients `ak` and `bk`. The new node inherits the constraints
 * from its parent node `p` and applies additional constraints to enforce integrality.
 *
 * @param p   Pointer to the parent `node_t` structure.
 * @param m   Number of constraints in the ILP problem.
 * @param n   Number of decision variables in the ILP problem.
 * @param a   2D array representing the constraint matrix (size m x n).
 * @param b   Array representing the right-hand side values of the constraints (size m).
 * @param c   Array representing the objective function coefficients (size n).
 * @param k   Index of the variable to branch on.
 * @param ak  Coefficient associated with the variable `xk` in the branching constraint.
 * @param bk  Right-hand side value for the branching constraint.
 *
 * @return A pointer to the newly created and extended `node_t` structure.
 */
node_t* extend(node_t* p, int m, int n, double** a, double* b, double* c, int k, double ak, double bk) {
    // Allocate memory for the new node and initialize it to zero
    node_t* q = calloc(1, sizeof(node_t));
    int i, j;

    // Set branching information
    q->k = k;
    q->ak = ak;
    q->bk = bk;

    // Determine the number of constraints for the new node based on branching direction
    if (ak > 0 && p->max[k] < INFINITY) {
        q->m = p->m;
    }
    else if (ak < 0 && p->min[k] > 0) {
        q->m = p->m;
    }
    else {
        q->m = p->m + 1; // Add an additional constraint if necessary
    }

    // Inherit the number of decision variables from the parent
    q->n = p->n;

    // Initialize the branch indicator
    q->h = -1;

    // Allocate memory for the constraint matrix (m+1 x n+1)
    q->a = calloc(q->m + 1, sizeof(double*));
    for (i = 0; i < q->m + 1; i++) {
        q->a[i] = calloc(q->n + 1, sizeof(double));
    }

    // Allocate memory for the right-hand side values, objective coefficients, and solution variables
    q->b = calloc(q->m + 1, sizeof(double));
    q->c = calloc(q->n + 1, sizeof(double));
    q->x = calloc(q->n + 1, sizeof(double));

    // Allocate memory for the variable bounds
    q->min = malloc(q->n * sizeof(double));
    q->max = malloc(q->n * sizeof(double));

    // Inherit variable bounds from the parent node
    for (i = 0; i < n; i++) {
        q->min[i] = p->min[i];
        q->max[i] = p->max[i];
    }

    // Copy the constraint matrix from input to the new node's constraint matrix
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            q->a[i][j] = a[i][j];
        }
    }

    // Copy the right-hand side values from input to the new node's RHS array
    for (i = 0; i < m; i++) {
        q->b[i] = b[i];
    }

    // Copy the objective function coefficients from input to the new node's objective array
    for (i = 0; i < n; i++) {
        q->c[i] = c[i];
    }

    // Update variable bounds based on the branching direction
    if (ak > 0) {
        if (q->max[k] == INFINITY || bk < q->max[k]) {
            q->max[k] = bk;
        }
    }
    else if (q->min[k] == -INFINITY || -bk > q->min[k]) {
        q->min[k] = -bk;
    }

    // Add additional constraints to enforce variable bounds
    for (i = m, j = 0; j < n; j++) {
        // Enforce lower bound if it exists
        if (q->min[j] > -INFINITY) {
            q->a[i][j] = -1.0;
            q->b[i] = -q->min[j];
            i++;
        }
        // Enforce upper bound if it exists
        if (q->max[j] < INFINITY) {
            q->a[i][j] = 1.0;
            q->b[i] = q->max[j];
            i++;
        }
    }

    return q;
}

/**
 * @brief Checks if a given variable is integer within a specified tolerance.
 *
 * This function determines whether the value pointed to by `xp` is effectively an integer,
 * considering a small threshold (`EPSILON`) to account for floating-point precision errors.
 * If the value is close enough to an integer, it rounds the value to the nearest integer
 * and updates the original variable. Otherwise, it leaves the variable unchanged.
 *
 * @param xp Pointer to the double value representing a variable in the ILP solution.
 *
 * @return Returns 1 if the variable is integer (within the defined tolerance), otherwise 0.
 */
int is_integer(double* xp) {
    // Dereference the pointer to obtain the variable's value
    double x = *xp;
    
    // Round the value to the nearest integer using the lround function
    double r = lround(x); // ISO C lround
    
    // Check if the absolute difference between the rounded value and the original value is within EPSILON
    if (fabs(r - x) < EPSILON) {
        *xp = r; // Update the original variable to the rounded integer value
        return 1; // Indicate that the variable is integer
    } else {
        return 0; // Indicate that the variable is not integer
    }
}

/**
 * @brief Verifies whether all decision variables in a node are integer.
 *
 * This function iterates through all decision variables in the given node `p` and checks
 * if each variable is integer using the `is_integer` function. If all variables are integer,
 * it returns 1. If any variable is found to be non-integer, it returns 0 immediately.
 *
 * @param p Pointer to the `node_t` structure representing a node in the Branch-and-Bound tree.
 *
 * @return Returns 1 if all decision variables in the node are integer, otherwise 0.
 */
int integer(node_t* p) {
    // Iterate through each decision variable in the node
    for (int i = 0; i < p->n; i++) {
        // Check if the i-th variable is integer
        if (!is_integer(&p->x[i])) {
            return 0; // If any variable is not integer, return 0
        }
    }
    return 1; // All variables are integer, return 1
}

/**
 * @brief Removes and returns the top node from the stack.
 *
 * This function pops the top node from the linked stack pointed to by `h`.
 * If the stack is empty, it returns `NULL`. Otherwise, it removes the top
 * node, updates the stack head, frees the temporary stack node structure,
 * and returns the popped `node_t` pointer.
 *
 * @param h Double pointer to the head of the linked stack (`linked_node_stack_t`).
 *
 * @return Pointer to the popped `node_t` structure, or `NULL` if the stack is empty.
 */
node_t* pop(linked_node_stack_t** h) {
    // Check if the stack is empty
    if (*h == NULL) {
        return NULL; // Empty stack
    }

    // Temporary pointer to hold the current head of the stack
    linked_node_stack_t* temp = *h;

    // Retrieve the node from the top of the stack
    node_t* popped_node = temp->node;

    // Update the head of the stack to the next node
    *h = temp->next;

    // Free the temporary stack node structure
    free(temp);

    // Return the popped node
    return popped_node;
}

/**
 * @brief Adds a node to the top of the stack.
 *
 * This function pushes a new `node_t` onto the linked stack pointed to by `h`.
 * It allocates memory for a new `linked_node_stack_t` structure, assigns the
 * provided `node_t` pointer to it, and updates the stack head to point to the
 * new node. If memory allocation fails, the function prints an error message
 * and exits the program.
 *
 * @param h    Double pointer to the head of the linked stack (`linked_node_stack_t`).
 * @param node Pointer to the `node_t` structure to be pushed onto the stack.
 */
void push(linked_node_stack_t** h, node_t* node) {
    // Allocate memory for the new stack node
    linked_node_stack_t* new_node = malloc(sizeof(linked_node_stack_t));
    if (new_node == NULL) {
        fprintf(stderr, "Error: Memory allocation failed in push().\n");
        exit(EXIT_FAILURE);
    }

    // Assign the node to the new stack node
    new_node->node = node;

    // Link the new stack node to the current head of the stack
    new_node->next = *h;

    // Update the head of the stack to point to the new node
    *h = new_node;
}

/**
 * @brief Frees all memory associated with a given node.
 *
 * This function deallocates all dynamically allocated memory within a `node_t`
 * structure, including the constraint matrix, right-hand side values, objective
 * coefficients, solution variables, and variable bounds. After freeing the
 * internal structures, it frees the `node_t` structure itself.
 *
 * @param q Pointer to the `node_t` structure to be freed.
 */
void free_node(node_t* q) {
    // Free the lower and upper bounds arrays
    free(q->min);
    free(q->max);

    // Free the right-hand side values array
    free(q->b);

    // Free the solution variables array
    free(q->x);

    // Free the objective function coefficients array
    free(q->c);

    // Free each row of the constraint matrix
    for (int i = 0; i < q->m + 1; i++) {
        free(q->a[i]);
    }

    // Free the constraint matrix itself
    free(q->a);

    // Free the node structure
    free(q);
}

/**
 * @brief Updates the upper bound and prunes the stack based on the current node's solution.
 *
 * This function compares the objective value `z` of the current node `p` with the best
 * objective value found so far (`*zp`). If `p->z` is greater, it updates `*zp` and the
 * solution vector `x` with the current node's solution. It then iterates through the
 * stack `h`, retaining only those nodes with an objective value greater than or equal to
 * `p->z` and discards the rest.
 *
 * @param p   Pointer to the current `node_t` structure being evaluated.
 * @param h   Double pointer to the head of the linked stack (`linked_node_stack_t`).
 * @param zp  Pointer to the current best (maximum) objective value found.
 * @param x   Pointer to the array storing the best solution variables found so far.
 */
void bound(node_t* p, linked_node_stack_t** h, double* zp, double* x) {
    // Check if the current node's objective value exceeds the best found so far
    if (p->z > *zp) {
        // Update the best objective value
        *zp = p->z;
        
        // Update the best solution vector with the current node's solution
        for (int i = 0; i < p->n + 1; i++) {
            x[i] = p->x[i];
        }
        
        // Temporary stack to hold nodes that meet the pruning criteria
        linked_node_stack_t* nodes_to_keep = NULL;
        node_t* q;
        
        // Iterate through the current stack and retain only nodes with z >= p->z
        while ((q = pop(h)) != NULL) {
            if (q->z >= p->z) {
                push(&nodes_to_keep, q); // Keep the node
            } else {
                free_node(q); // Discard the node
            }
        }
        
        // Update the original stack to only include the retained nodes
        *h = nodes_to_keep;
    }
}

/**
 * @brief Determines whether to branch on a node and selects the variable to branch on.
 *
 * This function evaluates whether the current node `q` should be branched based on its
 * objective value `z` compared to a given threshold `z`. If branching is necessary, it
 * identifies a non-integer variable to branch on, ensuring that the branching respects
 * the variable's bounds (`min` and `max`). The function updates the node's branching
 * variable index `h` and its value `xh` accordingly.
 *
 * @param q   Pointer to the `node_t` structure representing the node to evaluate for branching.
 * @param z   Threshold objective value to determine if branching should occur.
 *
 * @return Returns 1 if branching is performed (i.e., a non-integer variable is found and selected),
 *         otherwise returns 0 indicating no branching is necessary.
 */
int branch(node_t* q, double z) {
    double min, max;

    // If the node's objective value is less than the threshold, do not branch
    if (q->z < z) {
        return 0;
    }

    // Iterate through all decision variables to find a non-integer variable
    for (int h = 0; h < q->n; h++) {
        // Check if the current variable is not integer
        if (!is_integer(&q->x[h])) {
            // Determine the minimum bound for branching
            if (q->min[h] == 1) {
                min = 0;
            } else {
                min = q->min[h];
            }

            // Determine the maximum bound for branching
            max = q->max[h];

            // Check if the variable's floor or ceiling violates its bounds
            if (floor(q->x[h]) < min || ceil(q->x[h]) > max) {
                continue; // Skip branching on this variable
            }

            // Set the branching variable index and its current value
            q->h = h;
            q->xh = q->x[h];
            return 1; // Branching performed
        }
    }

    // No suitable non-integer variable found for branching
    return 0;
}

/**
 * @brief Processes a node in the Branch-and-Bound tree for Integer Linear Programming (ILP).
 *
 * This function extends the current node `p` by branching on a specific variable. It then
 * solves the resulting linear program using the simplex algorithm. Depending on whether
 * the solution is integer feasible, it either updates the best known solution and prunes
 * the stack or continues branching by adding new nodes to the stack. Memory management
 * is carefully handled to prevent leaks by freeing nodes that are no longer needed.
 *
 * @param p    Pointer to the current `node_t` structure being processed.
 * @param h    Double pointer to the head of the linked stack (`linked_node_stack_t`).
 * @param m    Number of constraints in the ILP problem.
 * @param n    Number of decision variables in the ILP problem.
 * @param a    2D array representing the constraint matrix (size m x n).
 * @param b    Array representing the right-hand side values of the constraints (size m).
 * @param c    Array representing the objective function coefficients (size n).
 * @param k    Index of the variable to branch on.
 * @param ak   Coefficient associated with the variable `xk` in the branching constraint.
 * @param bk   Right-hand side value for the branching constraint.
 * @param zp   Pointer to the current best (maximum) objective value found.
 * @param x    Pointer to the array storing the best solution variables found so far.
 */
void succ(node_t* p, linked_node_stack_t** h, int m, int n, double** a, double* b,
          double* c, int k, double ak, double bk, double* zp, double* x) {

    // Extend the current node by branching on variable k
    node_t* q = extend(p, m, n, a, b, c, k, ak, bk);
    if (q == NULL) {
        return; // Extension failed, possibly due to memory allocation issues
    }

    // Solve the linear program using the simplex algorithm and store the objective value in q->z
    q->z = simplex(q->m, q->n, q->a, q->b, q->c, q->x, 0);

    // Check if the solution is finite (i.e., the LP is feasible and bounded)
    if (isfinite(q->z)) {
        // THIS IS A BIT DIFFERENT FROM PSEUDOCODE, BUT TO HANDLE MEMORY LEAKS!!!

        if (integer(q)) {
            // Found an integer feasible solution
            bound(q, h, zp, x); // Update the best known solution and prune the stack

            // Since q was never pushed to the stack, free its memory to prevent leaks
            free_node(q);
        } else {
            // Solution is not integer feasible; attempt to branch further
            if (branch(q, *zp)) {
                // Branching is possible; push the node onto the stack for further exploration
                push(h, q);
            } else {
                // Branching is not possible or beneficial; free the node to prevent memory leaks
                free_node(q);
            }
        }

        // Early return as processing for this node is complete
        return;
    }

    // If the solution is not finite, free the node as it's either infeasible or unbounded
    free_node(q);
}

/**
 * @brief Solves the Integer Linear Programming (ILP) problem using the Branch-and-Bound method.
 *
 * This function initializes the Branch-and-Bound tree with the initial node, performs
 * the simplex algorithm to find feasible solutions, and iteratively explores branches
 * to find the optimal integer solution. It maintains the best integer solution found
 * so far (`z`) and updates the solution vector (`x`) accordingly. The function handles
 * memory management by properly freeing nodes that are no longer needed to prevent leaks.
 *
 * @param m Number of constraints in the ILP problem.
 * @param n Number of decision variables in the ILP problem.
 * @param a 2D array representing the constraint matrix (size m x n).
 * @param b Array representing the right-hand side values of the constraints (size m).
 * @param c Array representing the objective function coefficients (size n).
 * @param x Array to store the best solution variables found (size n + 1).
 *
 * @return The optimal objective function value as a double.
 *         Returns `NAN` if no feasible integer solution exists.
 */
double intopt(int m, int n, double** a, double* b, double* c, double* x) {
    // Initialize the initial node with the provided ILP parameters
    node_t* p = initial_node(m, n, a, b, c);
    
    // Initialize the stack for Branch-and-Bound with the initial node
    linked_node_stack_t* h = NULL;
    push(&h, p);
    
    // Initialize the best integer solution found so far to negative infinity
    double z = -INFINITY; // Best integer solution found so far
    
    // Solve the linear program for the initial node using the simplex algorithm
    p->z = simplex(p->m, p->n, p->a, p->b, p->c, p->x, 0);
    
    // Check if the initial solution is integer feasible or if the LP is infeasible/unbounded
    if (integer(p) || !isfinite(p->z)) {
        z = p->z; // Update the best objective value
        
        if (integer(p)) {
            // If the initial solution is integer feasible, update the solution vector
            for (int i = 0; i < p->n + 1; i++) {
                x[i] = p->x[i];
            }
        }
        
        // Pop the initial node from the stack and free its memory
        node_t* temp = pop(&h); // This should return p
        free_node(temp);        // Now safe to free p since it's no longer on h
        
        // Free any remaining nodes in the stack (if any)
        while ((p = pop(&h)) != NULL) {
            free_node(p);
        }
        
        return z; // Return the best objective value found
    }
    
    // Attempt to branch on the initial node
    branch(p, z);
    
    // Iterate through the stack until it's empty
    while ((p = pop(&h)) != NULL) {
        // Process the node by branching in the positive direction
        succ(p, &h, m, n, a, b, c, p->h, 1, floor(p->xh), &z, x);
        
        // Process the node by branching in the negative direction
        succ(p, &h, m, n, a, b, c, p->h, -1, -ceil(p->xh), &z, x);
        
        // Free the node after processing to prevent memory leaks
        free_node(p);
    }
    
    // Check if a feasible integer solution was found
    if (z == -INFINITY) {
        return NAN; // Return Not-a-Number if no solution was found
    } else {
        return z; // Return the optimal objective function value
    }
}

#ifdef INCLUDE_MAIN
/**
 * @brief Entry point for the Integer Linear Programming (ILP) solver.
 *
 * This function initializes the ILP problem by reading input parameters, allocating necessary
 * memory, and setting up the constraint matrix and objective function. It then executes the
 * Branch-and-Bound algorithm to solve the ILP and outputs the result. Finally, it frees all
 * allocated memory to prevent memory leaks.
 *
 * @return Returns 0 upon successful execution.
 */
int main() {
    // Variables for the linear program
    int m;              /**< Number of constraints */
    int n;              /**< Number of decision variables */
    double** a;         /**< Constraint matrix (size m x (n + 1)) */
    double* b;          /**< Right-hand side values (size m) */
    double* c;          /**< Objective function coefficients (size n) */
    double* x;          /**< Solution variables (size n + 1) */
    double y = 0;       /**< Objective function value */

    // Read the number of constraints and decision variables
    scanf("%d %d", &m, &n);

    // Allocate memory for the solution variables
    x = calloc(n + 1, sizeof(double));

    // Read and store the objective function coefficients
    c = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        scanf("%lf", &c[i]);
    }
    
    // Allocate memory for the constraint matrix
    a = malloc(m * sizeof(double*));
    for (int i = 0; i < m; i++) {
        a[i] = malloc((n + 1) * sizeof(double));
        for (int j = 0; j < n; j++) {
            scanf("%lf", &a[i][j]);
        }
        a[i][n] = 0; // Initialize the additional parameter (e.g., slack variable coefficient) to zero
    }

    // Allocate and read the right-hand side values
    b = malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) {
        scanf("%lf", &b[i]);
    }

    // Execute the simplex algorithm using the Branch-and-Bound method
    double result = intopt(m, n, a, b, c, x);
    printf("Result: %lf\n", result);

    // Free allocated memory for the constraint matrix
    if (a != NULL) {
        for (int i = 0; i < m; i++) {
            free(a[i]);
        }
        free(a);
    }

    // Free other allocated memory
    free(b);
    free(c);
    free(x);

    return 0; // Indicate successful execution
}
#endif