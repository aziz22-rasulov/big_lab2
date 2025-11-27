import streamlit as st
import numpy as np
import time
from numpy.linalg import norm

st.set_page_config(page_title="Схема Халецкого", page_icon="🧮", layout="centered")

def haltsky_decomposition(A):
    """Разложение A = B*C по формулам из учебника"""
    n = len(A)
    B = np.zeros((n, n))
    C = np.zeros((n, n))
    
    for i in range(n):
        C[i, i] = 1.0
    
    for j in range(n):
        for i in range(j, n):
            if j == 0:
                B[i, j] = A[i, j]
            else:
                sum_val = 0.0
                for k in range(j):
                    sum_val += B[i, k] * C[k, j]
                B[i, j] = A[i, j] - sum_val
            
            if i == j and abs(B[i, j]) < 1e-10:
                raise ValueError(f"Элемент B[{i+1}][{j+1}] = {B[i, j]:.4e} близок к нулю")
        
        for i in range(j+1, n):
            sum_val = 0.0
            for k in range(j):
                sum_val += B[j, k] * C[k, i]
            C[j, i] = (A[j, i] - sum_val) / B[j, j]
    
    return B, C

def haltsky_solve(A, b):
    """Решение системы Ax = b методом Халецкого"""
    n = len(A)
    B, C = haltsky_decomposition(A)
    
    # Прямой ход: By = b
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += B[i, j] * y[j]
        y[i] = (b[i] - sum_val) / B[i, i]
    
    # Обратный ход: Cx = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = 0.0
        for j in range(i+1, n):
            sum_val += C[i, j] * x[j]
        x[i] = y[i] - sum_val
    
    return x, B, C

def verify_solution(A, b, x):
    """Проверка правильности решения"""
    Ax = A @ x
    residual = norm(Ax - b)
    relative_residual = residual / norm(b)
    return Ax, residual, relative_residual

def generate_test_matrix(n):
    """Генерация матрицы, удовлетворяющей условиям метода Халецкого"""
    # Создаем нижнюю треугольную матрицу B с ненулевой диагональю
    B = np.zeros((n, n))
    for i in range(n):
        B[i, i] = i + 1
        for j in range(i):
            B[i, j] = np.random.uniform(-5, 5)
    
    # Создаем верхнюю треугольную матрицу C с единицами на диагонали
    C = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            C[i, j] = np.random.uniform(-5, 5)
    
    # Формируем матрицу A = B * C
    A = B @ C
    b = np.random.uniform(-10, 10, n)
    return A, b

def main():
    st.title("🧮 Схема Халецкого")
    st.markdown("### Решение систем линейных уравнений")
    
    mode = st.radio("Выберите режим", ["Ручной ввод", "Сгенерировать систему (n≥50)"], horizontal=True)
    
    if mode == "Сгенерировать систему (n≥50)":
        n = st.slider("Размер системы", min_value=50, max_value=100, value=50)
        if st.button("Сгенерировать и решить", type="primary"):
            with st.spinner("Генерация и решение системы..."):
                A, b = generate_test_matrix(n)
                start_time = time.time()
                x, B, C = haltsky_solve(A, b)
                exec_time = time.time() - start_time
                Ax, residual, rel_residual = verify_solution(A, b, x)
            
            st.success("✅ Система успешно решена!")
            st.markdown(f"**Время решения:** {exec_time:.6f} сек")
            st.markdown(f"**Относительная невязка:** {rel_residual:.2e}")
            
            # Вывод решения
            st.markdown("### Решение системы:")
            for i in range(min(5, n)):  # Показываем только первые 5 переменных для больших n
                st.markdown(f"x<sub>{i+1}</sub> = {x[i]:.6f}", unsafe_allow_html=True)
            if n > 5:
                st.markdown(f"... и еще {n-5} переменных")
            
            # Проверка подстановкой
            st.markdown("### Проверка для первых 3 уравнений:")
            for i in range(min(3, n)):
                st.markdown(f"""
                Уравнение {i+1}:  
                Левая часть = {Ax[i]:.6f}  
                Правая часть = {b[i]:.6f}  
                Разница = {Ax[i] - b[i]:.2e}
                """)
    
    else:  # Ручной ввод
        n = st.number_input("Размер системы", min_value=2, max_value=6, value=3)
        
        st.markdown("### Введите коэффициенты системы:")
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for i in range(n):
            cols = st.columns(n + 1)
            for j in range(n):
                A[i, j] = cols[j].number_input(f"a{i+1}{j+1}", value=0.0, key=f"a_{i}_{j}", step=1.0)
            b[i] = cols[n].number_input(f"b{i+1}", value=0.0, key=f"b_{i}", step=1.0)
        
        if st.button("Решить систему", type="primary"):
            try:
                # Проверка условий применимости
                try:
                    B_test, C_test = haltsky_decomposition(A)
                    st.success("✅ Условия применимости метода выполнены")
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
                    st.stop()
                
                # Решение системы
                x, B, C = haltsky_solve(A, b)
                Ax, residual, rel_residual = verify_solution(A, b, x)
                
                # Вывод результатов
                st.markdown("### Результаты решения:")
                st.markdown(f"**Относительная невязка:** {rel_residual:.2e}")
                
                st.markdown("#### Вектор решения x:")
                for i in range(n):
                    st.markdown(f"x<sub>{i+1}</sub> = {x[i]:.6f}", unsafe_allow_html=True)
                
                st.markdown("#### Проверка подстановкой:")
                for i in range(n):
                    st.markdown(f"""
                    Уравнение {i+1}:  
                    ∑a<sub>{i+1}j</sub>x<sub>j</sub> = {Ax[i]:.6f},  b<sub>{i+1}</sub> = {b[i]:.6f},  
                    Разница = {Ax[i] - b[i]:.2e}
                    """, unsafe_allow_html=True)
                
                # Исследование скорости (для сравнения)
                if n >= 3:
                    st.markdown("### Скорость работы метода:")
                    start_time = time.time()
                    x_gauss = np.linalg.solve(A, b)
                    gauss_time = time.time() - start_time
                    halt_time = time.time() - start_time - gauss_time
                    
                    st.markdown(f"Метод Халецкого: {halt_time:.6f} сек")
                    st.markdown(f"Метод Гаусса (встроенная функция): {gauss_time:.6f} сек")
            
            except Exception as e:
                st.error(f"❌ Ошибка при решении: {str(e)}")

if __name__ == "__main__":
    main()
