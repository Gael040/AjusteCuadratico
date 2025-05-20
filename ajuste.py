import streamlit as st
import numpy as np
from sympy import sympify, symbols, lambdify
import matplotlib.pyplot as plt

# Función para evaluar la expresión
def evaluar(expr, var, val):
    f = lambdify(var, expr, 'numpy')
    return f(val)

# Streamlit UI
st.title("Ajuste cuadrático")
user_input = st.text_input("Ingresa una ecuación de una sola variable (e.g., x**2 - 4*x + 5):")

if user_input:
    try:
        e = sympify(user_input)
        variables = list(e.free_symbols)
        
        if len(variables) != 1:
            st.error("Ingresa una ecuación de UNA sola variable.")
        else:
            x = variables[0]

            # Genera 3 puntos aleatorios
            puntos_inciales = np.random.randint(0, 10, 3).tolist()
            st.write("Puntos iniciales:", puntos_inciales)

            for iter in range(10):
                st.subheader(f"Iteración {iter + 1}")

                if iter == 0:
                    evaluados = [evaluar(e, x, i) for i in puntos_inciales]

                # Matriz A
                a = []
                for i in puntos_inciales:
                    a.extend([i**2, i, 1])
                a = np.array(a).reshape(3, 3)

                b = np.array(evaluados)

                st.write("Matriz A:")
                st.write(a)
                st.write("Vector b (evaluaciones):")
                st.write(b)

                try:
                    coeficientes = np.linalg.solve(a, b)
                except np.linalg.LinAlgError:
                    st.error("La matriz no tiene inversa.")
                    index = evaluados.index(min(evaluados))
                    st.write(f"Mejor x encontrada: {puntos_inciales[index]}")
                    st.write(f"Función evaluada en mejor x: {evaluados[index]}")
                    break

                # Vertex of the parabola: x = -b / (2a)
                a_coef, b_coef = coeficientes[0], coeficientes[1]
                x_min = -b_coef / (2 * a_coef)
                x_min_eval = evaluar(e, x, x_min)

                # Replace worst point if new point is better
                max_idx = evaluados.index(max(evaluados))
                if x_min_eval < evaluados[max_idx]:
                    evaluados[max_idx] = float(x_min_eval)
                    puntos_inciales[max_idx] = x_min

                st.write(f"Nuevo candidato x_min: {x_min}")
                st.write(f"Función evaluada en x_min: {x_min_eval}")
                st.write("Puntos actualizados:", puntos_inciales)
                st.write("Evaluaciones actualizadas:", evaluados)
                st.markdown("---")

            # Final result
            best_idx = evaluados.index(min(evaluados))
            st.success(f"Optimización completada. Mejor x: {puntos_inciales[best_idx]}, Value: {evaluados[best_idx]}")

    except Exception as ex:
        st.error(f"Error: {str(ex)}")

# After the optimization is done:
if user_input and len(variables) == 1:
    # Define range for x-axis around your evaluated points
    x_vals = np.linspace(min(puntos_inciales) - 5, max(puntos_inciales) + 5, 400)
    y_vals = [evaluar(e, x, val) for val in x_vals]

    # Final best result
    best_idx = evaluados.index(min(evaluados))
    x_best = puntos_inciales[best_idx]
    y_best = evaluados[best_idx]

    # Plot
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="Función", color="blue")
    ax.scatter(puntos_inciales, [evaluar(e, x, val) for val in puntos_inciales], color='orange', label="Points")
    ax.scatter(x_best, y_best, color='red', label="Best Found", zorder=5)
    ax.axvline(x_best, color='red', linestyle='--', alpha=0.6)
    ax.set_title("Función y proceso de optimización")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
