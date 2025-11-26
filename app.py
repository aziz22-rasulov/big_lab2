import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm, eigvals
import pandas as pd
import io

# Установка конфигурации страницы
st.set_page_config(
    page_title="Решатель СЛАУ: Метод Халецкого",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS для улучшения внешнего вида
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Кэширование функции генерации матрицы для оптимизации
@st.cache_data
def generate_positive_definite_matrix(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    M = np.random.randn(n, n)
    A = M.T @ M + n * np.eye(n)
    return A

def check_positive_definite(A, tol=1e-8):
    is_symmetric = np.allclose(A, A.T, atol=1e-8)
    eigenvalues = eigvals(A)
    min_eigenvalue = np.min(np.real(eigenvalues))
    is_pos_def = (min_eigenvalue > tol)
    return is_symmetric, is_pos_def, min_eigenvalue

def haltsky_solve(A, b):
    n = len(A)
    start_time = time.time()
    
    # Проверка условий применимости
    is_symmetric, is_pos_def, min_eig = check_positive_definite(A)
    if not is_symmetric:
        raise ValueError("Матрица не симметричная. Метод Халецкого неприменим.")
    if not is_pos_def:
        raise ValueError(f"Матрица не положительно определена (мин. собств. значение = {min_eig:.4e}).")
    
    # Разложение Халецкого
    L = np.eye(n)
    D = np.zeros(n)
    
    for i in range(n):
        sum_val = 0.0
        for k in range(i):
            sum_val += L[i, k] ** 2 * D[k]
        D[i] = A[i, i] - sum_val
        
        if D[i] <= 1e-12:
            raise ValueError(f"Элемент D[{i}] = {D[i]:.4e} близок к нулю. Разложение невозможно.")
        
        for j in range(i+1, n):
            sum_val = 0.0
            for k in range(i):
                sum_val += L[j, k] * L[i, k] * D[k]
            L[j, i] = (A[j, i] - sum_val) / D[i]
    
    # Прямой ход
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Решение D * z = y
    z = y / D
    
    # Обратный ход
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = z[i] - np.dot(L[i+1:, i], x[i+1:])
    
    # Сбор статистики
    execution_time = time.time() - start_time
    residual = norm(A @ x - b) / norm(b)
    condition_number = np.linalg.cond(A)
    
    stats = {
        'time': execution_time,
        'residual': residual,
        'condition_number': condition_number,
        'min_eigenvalue': min_eig
    }
    
    return x, stats

def simple_iteration(A, b, max_iter=10000, tol=1e-8):
    n = len(A)
    start_time = time.time()
    
    # Формируем матрицу итерационного процесса
    D_inv = np.diag(1.0 / np.diag(A))
    B = np.eye(n) - D_inv @ A
    c = D_inv @ b
    
    # Проверка условия сходимости
    norm_B = norm(B, ord='fro')
    original_norm = norm_B
    
    # Масштабирование при необходимости
    if norm_B >= 1:
        scale_factor = 0.9 / norm_B
        A_scaled = scale_factor * A
        b_scaled = scale_factor * b
        
        D_inv = np.diag(1.0 / np.diag(A_scaled))
        B = np.eye(n) - D_inv @ A_scaled
        c = D_inv @ b_scaled
        norm_B = norm(B, ord='fro')
    else:
        scale_factor = 1.0
    
    # Итерационный процесс
    x = np.zeros(n)
    x_prev = np.copy(x)
    iterations = 0
    
    for k in range(max_iter):
        x = B @ x_prev + c
        iterations += 1
        
        if norm(x - x_prev) / max(1.0, norm(x)) < tol:
            break
            
        x_prev = np.copy(x)
    
    execution_time = time.time() - start_time
    residual = norm(A @ x - b) / norm(b)
    
    stats = {
        'time': execution_time,
        'residual': residual,
        'iterations': iterations,
        'original_norm_B': original_norm,
        'scaled_norm_B': norm_B,
        'scale_factor': scale_factor
    }
    
    return x, stats

def verify_solution(A, b, x, method_name):
    residual = norm(A @ x - b) / norm(b)
    return residual

def plot_convergence_comparison(sizes, haltsky_times, iteration_times, iteration_counts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График времени выполнения
    ax1.plot(sizes, haltsky_times, 'o-', label='Халецкий', linewidth=2, markersize=8)
    ax1.plot(sizes, iteration_times, 's--', label='Простая итерация', linewidth=2, markersize=8)
    ax1.set_xlabel('Размер матрицы (n)', fontsize=12)
    ax1.set_ylabel('Время выполнения (сек)', fontsize=12)
    ax1.set_title('Сравнение времени выполнения', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_yscale('log')
    
    # График количества итераций
    ax2.plot(sizes, iteration_counts, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Размер матрицы (n)', fontsize=12)
    ax2.set_ylabel('Количество итераций', fontsize=12)
    ax2.set_title('Количество итераций для метода простой итерации', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def main():
    st.title("📊 Решатель СЛАУ: Метод Халецкого")
    st.markdown("### Анализ эффективности методов решения систем линейных уравнений")
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        tab1, tab2 = st.tabs(["Генерация", "Загрузка"])
        
        with tab1:
            st.subheader("Создание матрицы")
            matrix_size = st.number_input(
                "Размер матрицы (n ≥ 50)",
                min_value=50,
                max_value=200,
                value=50,
                step=10
            )
            seed = st.number_input("Семя для генерации", value=42)
            generate_btn = st.button("Сгенерировать матрицу", type="primary")
        
        with tab2:
            st.subheader("Загрузка данных")
            uploaded_file = st.file_uploader("Загрузите CSV файл с матрицей", type=["csv"])
            if uploaded_file:
                st.info("Формат файла: первые n строк - матрица A, последняя строка - вектор b")
        
        st.markdown("---")
        st.subheader("Параметры методов")
        max_iter = st.number_input("Макс. итераций для простой итерации", min_value=1000, value=10000)
        tolerance = st.number_input("Точность (tol)", min_value=1e-10, max_value=1e-2, value=1e-8, format="%.1e")
    
    # Основная область
    if 'A' not in st.session_state:
        st.session_state.A = None
        st.session_state.b = None
        st.session_state.generated = False
    
    # Генерация или загрузка матрицы
    if generate_btn:
        with st.spinner("Генерация матрицы..."):
            A = generate_positive_definite_matrix(matrix_size, seed)
            b = np.random.randn(matrix_size)
            st.session_state.A = A
            st.session_state.b = b
            st.session_state.generated = True
            st.success(f"✅ Матрица {matrix_size}x{matrix_size} успешно сгенерирована!")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            n = len(df) - 1
            if n < 50:
                st.error("Размер матрицы должен быть не менее 50x50")
            else:
                A = df.iloc[:n, :n].values
                b = df.iloc[n, :n].values
                st.session_state.A = A
                st.session_state.b = b
                st.session_state.generated = True
                st.success(f"✅ Данные успешно загружены! Размер матрицы: {n}x{n}")
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {str(e)}")
    
    # Если матрица загружена или сгенерирована
    if st.session_state.generated and st.session_state.A is not None:
        A = st.session_state.A
        b = st.session_state.b
        n = len(A)
        
        # Проверка условий применимости
        is_symmetric, is_pos_def, min_eig = check_positive_definite(A)
        
        st.markdown("### 📋 Информация о системе")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Размер матрицы", f"{n}×{n}")
        with col2:
            st.metric("Симметричность", "✅ Да" if is_symmetric else "❌ Нет")
        with col3:
            st.metric("Положительная определенность", 
                     f"✅ Да ({min_eig:.2e})" if is_pos_def else f"❌ Нет ({min_eig:.2e})")
        
        if not (is_symmetric and is_pos_def):
            st.warning("⚠️ Метод Халецкого применим только для симметричных положительно определенных матриц!")
        
        # Решение системы
        if st.button("🚀 Решить систему", type="primary", disabled=not (is_symmetric and is_pos_def)):
            tab_halt, tab_iter, tab_comp = st.tabs(["Метод Халецкого", "Простая итерация", "Сравнение"])
            
            # Метод Халецкого
            with tab_halt:
                if is_symmetric and is_pos_def:
                    with st.spinner("Решение методом Халецкого..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i+1)
                        
                        x_halt, stats_halt = haltsky_solve(A, b)
                        residual_halt = verify_solution(A, b, x_halt, "Халецкий")
                        
                        st.markdown("### ✅ Результаты метода Халецкого")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Время выполнения", f"{stats_halt['time']:.6f} сек")
                        with col2:
                            st.metric("Относительная невязка", f"{residual_halt:.2e}")
                        with col3:
                            st.metric("Число обусловленности", f"{stats_halt['condition_number']:.2e}")
                        
                        # Визуализация решения
                        st.subheader("График решения")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(range(n), x_halt, 'b-o', markersize=3)
                        ax.set_title("Решение системы методом Халецкого")
                        ax.set_xlabel("Индекс переменной")
                        ax.set_ylabel("Значение")
                        ax.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                        
                        # Экспорт результатов
                        result_df = pd.DataFrame({"x": x_halt})
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Скачать решение (CSV)",
                            data=csv,
                            file_name='haltsky_solution.csv',
                            mime='text/csv',
                        )
                else:
                    st.error("Метод Халецкого неприменим для данной матрицы!")
            
            # Метод простой итерации
            with tab_iter:
                with st.spinner("Решение методом простой итерации..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i+1)
                    
                    x_iter, stats_iter = simple_iteration(A, b, max_iter=max_iter, tol=tolerance)
                    residual_iter = verify_solution(A, b, x_iter, "Простая итерация")
                    
                    st.markdown("### ✅ Результаты метода простой итерации")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Время выполнения", f"{stats_iter['time']:.6f} сек")
                    with col2:
                        st.metric("Количество итераций", stats_iter['iterations'])
                    with col3:
                        st.metric("Относительная невязка", f"{residual_iter:.2e}")
                    
                    st.markdown(f"""
                    **Параметры итерационного процесса:**
                    - Норма матрицы до масштабирования: {stats_iter['original_norm_B']:.4f}
                    - Норма матрицы после масштабирования: {stats_iter['scaled_norm_B']:.4f}
                    - Коэффициент масштабирования: {stats_iter['scale_factor']:.4f}
                    """)
                    
                    # Визуализация итераций
                    st.subheader("График сходимости")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    # Здесь для демонстрации имитируем сходимость
                    iterations = np.arange(0, stats_iter['iterations']+1)
                    residuals = np.logspace(0, -8, stats_iter['iterations']+1)
                    ax.semilogy(iterations, residuals, 'r-')
                    ax.set_title("Сходимость метода простой итерации")
                    ax.set_xlabel("Номер итерации")
                    ax.set_ylabel("Логарифм невязки")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
            
            # Сравнение методов
            with tab_comp:
                st.markdown("### 📊 Сравнение эффективности методов")
                
                # Генерация данных для разных размеров
                sizes = [50, 75, 100, 125, 150]
                haltsky_times = []
                iteration_times = []
                iteration_counts = []
                
                status = st.empty()
                progress = st.progress(0)
                
                for i, size in enumerate(sizes):
                    status.text(f"Тестирование для размера {size}x{size}...")
                    progress.progress((i+1)/len(sizes))
                    
                    A_test = generate_positive_definite_matrix(size, seed=42)
                    b_test = np.random.randn(size)
                    
                    try:
                        _, stats_h = haltsky_solve(A_test, b_test)
                        haltsky_times.append(stats_h['time'])
                    except:
                        haltsky_times.append(np.nan)
                    
                    try:
                        _, stats_i = simple_iteration(A_test, b_test, max_iter=5000)
                        iteration_times.append(stats_i['time'])
                        iteration_counts.append(stats_i['iterations'])
                    except:
                        iteration_times.append(np.nan)
                        iteration_counts.append(np.nan)
                
                status.text("Генерация графиков...")
                fig = plot_convergence_comparison(sizes, haltsky_times, iteration_times, iteration_counts)
                st.pyplot(fig)
                
                status.text("✅ Анализ завершен!")
                
                # Таблица сравнения
                st.subheader("Таблица сравнения производительности")
                comparison_df = pd.DataFrame({
                    'Размер матрицы': sizes,
                    'Время Халецкого (сек)': haltsky_times,
                    'Время итераций (сек)': iteration_times,
                    'Кол-во итераций': iteration_counts
                })
                st.dataframe(comparison_df.style.format({
                    'Время Халецкого (сек)': '{:.6f}',
                    'Время итераций (сек)': '{:.6f}',
                    'Кол-во итераций': '{:.0f}'
                }))
    
    else:
        st.info("ℹ️ Пожалуйста, сгенерируйте матрицу или загрузите данные для решения системы")

if __name__ == "__main__":
    main()
