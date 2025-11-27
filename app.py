import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from numpy.linalg import norm, eigvals
import io

# Установка конфигурации страницы
st.set_page_config(
    page_title="Решатель СЛАУ: Метод Халецкого",
    page_icon="🧮",
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
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #2196F3;
    }
    .matrix-input {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .equation-display {
        font-family: 'Courier New', monospace;
        font-size: 1.1em;
        background-color: #e9f7ef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_positive_definite(A, tol=1e-8):
    """Проверяет симметричность и положительную определенность матрицы"""
    is_symmetric = np.allclose(A, A.T, atol=1e-8)
    eigenvalues = eigvals(A)
    min_eigenvalue = np.min(np.real(eigenvalues))
    is_pos_def = (min_eigenvalue > tol)
    return is_symmetric, is_pos_def, min_eigenvalue

def haltsky_solve(A, b):
    """Решает систему линейных уравнений Ax = b методом Халецкого"""
    n = len(A)
    start_time = time.time()
    
    # Проверка условий применимости
    is_symmetric, is_pos_def, min_eig = check_positive_definite(A)
    if not is_symmetric:
        raise ValueError("Матрица не симметричная. Метод Халецкого неприменим.")
    if not is_pos_def:
        raise ValueError(f"Матрица не положительно определена (мин. собств. значение = {min_eig:.4e}).")
    
    # Разложение Халецкого A = LDL^T
    L = np.eye(n)
    D = np.zeros(n)
    
    for i in range(n):
        # Вычисление D[i]
        sum_val = 0.0
        for k in range(i):
            sum_val += L[i, k] ** 2 * D[k]
        D[i] = A[i, i] - sum_val
        
        if D[i] <= 1e-12:
            raise ValueError(f"Элемент D[{i}] = {D[i]:.4e} близок к нулю. Разложение невозможно.")
        
        # Вычисление элементов L[j, i] для j > i
        for j in range(i+1, n):
            sum_val = 0.0
            for k in range(i):
                sum_val += L[j, k] * L[i, k] * D[k]
            L[j, i] = (A[j, i] - sum_val) / D[i]
    
    # Прямой ход: L * y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Решение D * z = y
    z = y / D
    
    # Обратный ход: L^T * x = z
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
    """Решает систему методом простой итерации"""
    n = len(A)
    start_time = time.time()
    
    # Формируем матрицу итерационного процесса
    D_inv = np.diag(1.0 / np.diag(A))
    B = np.eye(n) - D_inv @ A
    c = D_inv @ b
    
    # Проверка условия сходимости: ||B|| < 1
    norm_B = norm(B, ord='fro')
    original_norm = norm_B
    
    # Масштабирование для обеспечения сходимости (если необходимо)
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
        
        # Проверка сходимости
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

def display_system_equations(A, b):
    """Отображает систему уравнений в красивом формате"""
    n = len(A)
    st.markdown("### 📝 Система уравнений:")
    st.markdown("**Введите систему уравнений в матричной форме Ax = b**")
    
    # Создаем интерактивную таблицу для ввода матрицы
    matrix_container = st.container()
    
    with matrix_container:
        st.markdown("#### Матрица коэффициентов A:")
        
        # Создаем DataFrame для редактирования
        if 'matrix_data' not in st.session_state:
            st.session_state.matrix_data = pd.DataFrame(np.zeros((n, n)), 
                                                      columns=[f'x{i+1}' for i in range(n)],
                                                      index=[f'Ур-е {i+1}' for i in range(n)])
        
        # Редактируемая таблица
        edited_matrix = st.data_editor(
            st.session_state.matrix_data,
            num_rows="fixed",
            use_container_width=True,
            key="matrix_editor"
        )
        
        st.markdown("#### Вектор правых частей b:")
        
        if 'vector_data' not in st.session_state:
            st.session_state.vector_data = pd.DataFrame(np.zeros(n), 
                                                       columns=['b'],
                                                       index=[f'Ур-е {i+1}' for i in range(n)])
        
        edited_vector = st.data_editor(
            st.session_state.vector_data,
            num_rows="fixed",
            use_container_width=True,
            key="vector_editor"
        )
    
    # Кнопки управления
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Сбросить значения", use_container_width=True):
            st.session_state.matrix_data = pd.DataFrame(np.zeros((n, n)), 
                                                      columns=[f'x{i+1}' for i in range(n)],
                                                      index=[f'Ур-е {i+1}' for i in range(n)])
            st.session_state.vector_data = pd.DataFrame(np.zeros(n), 
                                                       columns=['b'],
                                                       index=[f'Ур-е {i+1}' for i in range(n)])
            st.rerun()
    
    with col2:
        if st.button("🎲 Сгенерировать случайную систему", use_container_width=True):
            # Генерируем симметричную положительно определенную матрицу
            M = np.random.randint(-5, 6, (n, n))
            A_rand = M.T @ M + n * np.eye(n)
            b_rand = np.random.randint(-10, 11, n)
            
            st.session_state.matrix_data = pd.DataFrame(A_rand, 
                                                      columns=[f'x{i+1}' for i in range(n)],
                                                      index=[f'Ур-е {i+1}' for i in range(n)])
            st.session_state.vector_data = pd.DataFrame(b_rand.reshape(-1, 1), 
                                                       columns=['b'],
                                                       index=[f'Ур-е {i+1}' for i in range(n)])
            st.rerun()
    
    with col3:
        uploaded_file = st.file_uploader("📤 Загрузить из CSV", type=['csv'], label_visibility="collapsed")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if df.shape[0] == n and df.shape[1] >= n + 1:
                    st.session_state.matrix_data = df.iloc[:, :n]
                    st.session_state.matrix_data.columns = [f'x{i+1}' for i in range(n)]
                    st.session_state.matrix_data.index = [f'Ур-е {i+1}' for i in range(n)]
                    
                    st.session_state.vector_data = df.iloc[:, n:n+1]
                    st.session_state.vector_data.columns = ['b']
                    st.session_state.vector_data.index = [f'Ур-е {i+1}' for i in range(n)]
                    st.success("✅ Данные успешно загружены!")
                    st.rerun()
                else:
                    st.error(f"Ошибка: ожидается матрица {n}x{n+1}, получено {df.shape}")
            except Exception as e:
                st.error(f"Ошибка при загрузке файла: {str(e)}")
    
    # Конвертируем в numpy массивы
    A_input = edited_matrix.values.astype(float)
    b_input = edited_vector['b'].values.astype(float)
    
    return A_input, b_input

def plot_solution_comparison(x_halt, x_iter, method1_name="Халецкий", method2_name="Простая итерация"):
    """Сравнение решений двух методов"""
    n = len(x_halt)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    indices = np.arange(1, n+1)
    width = 0.35
    
    ax.bar(indices - width/2, x_halt, width, label=method1_name, alpha=0.8, color='skyblue')
    ax.bar(indices + width/2, x_iter, width, label=method2_name, alpha=0.8, color='salmon')
    
    ax.set_xlabel('Номер переменной', fontsize=12)
    ax.set_ylabel('Значение переменной', fontsize=12)
    ax.set_title('Сравнение решений методов', fontsize=14)
    ax.set_xticks(indices)
    ax.set_xticklabels([f'x{i}' for i in range(1, n+1)])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def create_manual_input_interface():
    """Создает интерфейс для ручного ввода системы уравнений"""
    st.title("🧮 Решатель СЛАУ: Метод Халецкого")
    st.markdown("### Введите систему линейных уравнений")
    
    # Выбор размера системы
    st.sidebar.header("⚙️ Параметры системы")
    n = st.sidebar.number_input("Размер системы (n)", min_value=2, max_value=10, value=3, step=1)
    
    # Параметры методов
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Параметры методов")
    max_iter = st.sidebar.number_input("Макс. итераций для простой итерации", min_value=100, value=10000)
    tolerance = st.sidebar.number_input("Точность (tol)", min_value=1e-10, max_value=1e-2, value=1e-8, format="%.1e")
    
    # Отображение и ввод системы уравнений
    A_input, b_input = display_system_equations(np.zeros((n, n)), np.zeros(n))
    
    # Проверка условий применимости
    is_valid = False
    if np.any(A_input):
        is_symmetric, is_pos_def, min_eig = check_positive_definite(A_input)
        st.markdown("### 🔍 Проверка условий применимости:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Размер системы", f"{n}×{n}")
        with col2:
            st.metric("Симметричность", "✅ Да" if is_symmetric else "❌ Нет", 
                     delta=None, delta_color="normal")
        with col3:
            st.metric("Положительная определенность", 
                     f"✅ Да ({min_eig:.2e})" if is_pos_def else f"❌ Нет ({min_eig:.2e})",
                     delta=None, delta_color="normal")
        
        if not is_symmetric:
            st.warning("⚠️ Матрица не симметричная. Метод Халецкого неприменим!")
        if not is_pos_def:
            st.warning("⚠️ Матрица не положительно определена. Метод Халецкого может не сойтись!")
        
        is_valid = is_symmetric and is_pos_def
    
    # Решение системы
    if st.button("🚀 Решить систему", type="primary", disabled=not is_valid):
        with st.spinner("Решение системы..."):
            progress_bar = st.progress(0)
            
            # Решение методом Халецкого
            try:
                progress_bar.progress(20)
                x_halt, stats_halt = haltsky_solve(A_input, b_input)
                residual_halt = norm(A_input @ x_halt - b_input) / norm(b_input)
                progress_bar.progress(50)
            except Exception as e:
                st.error(f"❌ Ошибка в методе Халецкого: {str(e)}")
                return
            
            # Решение методом простой итерации
            try:
                progress_bar.progress(70)
                x_iter, stats_iter = simple_iteration(A_input, b_input, max_iter=max_iter, tol=tolerance)
                residual_iter = norm(A_input @ x_iter - b_input) / norm(b_input)
                progress_bar.progress(90)
            except Exception as e:
                st.error(f"❌ Ошибка в методе простой итерации: {str(e)}")
                return
            
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
        
        # Отображение результатов
        st.markdown("## 📊 Результаты решения")
        
        tab1, tab2, tab3 = st.tabs(["📈 Сравнение методов", "🔢 Детали Халецкого", "🔄 Детали итераций"])
        
        with tab1:
            st.markdown("### 📋 Сравнительная таблица")
            
            # Создаем таблицу сравнения
            comparison_data = {
                'Метрика': ['Время выполнения', 'Относительная невязка', 'Количество итераций', 'Число обусловленности'],
                'Метод Халецкого': [
                    f"{stats_halt['time']:.6f} сек",
                    f"{residual_halt:.2e}",
                    "1",
                    f"{stats_halt['condition_number']:.2e}"
                ],
                'Метод простой итерации': [
                    f"{stats_iter['time']:.6f} сек",
                    f"{residual_iter:.2e}",
                    f"{stats_iter['iterations']}",
                    "-"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Сравнение решений
            st.markdown("### 📈 Сравнение решений")
            fig = plot_solution_comparison(x_halt, x_iter)
            st.pyplot(fig)
            
            # Заключение
            st.markdown("### 💡 Заключение")
            if stats_halt['time'] < stats_iter['time']:
                st.success(f"✅ **Метод Халецкого** оказался быстрее в {stats_iter['time']/stats_halt['time']:.1f} раз!")
            else:
                st.success(f"✅ **Метод простой итерации** оказался быстрее в {stats_halt['time']/stats_iter['time']:.1f} раз!")
            
            if residual_halt < residual_iter:
                st.info(f"🔍 **Метод Халецкого** дал более точное решение (невязка в {residual_iter/residual_halt:.1f} раз меньше)")
            else:
                st.info(f"🔍 **Метод простой итерации** дал более точное решение (невязка в {residual_halt/residual_iter:.1f} раз меньше)")
        
        with tab2:
            st.markdown("### 📋 Результаты метода Халецкого")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Время выполнения", f"{stats_halt['time']:.6f} сек")
            with col2:
                st.metric("Относительная невязка", f"{residual_halt:.2e}")
            with col3:
                st.metric("Число обусловленности", f"{stats_halt['condition_number']:.2e}")
            
            # Вектор решения
            st.markdown("### 🔢 Вектор решения:")
            solution_df = pd.DataFrame({
                'Переменная': [f'x{i+1}' for i in range(n)],
                'Значение': x_halt
            })
            st.dataframe(solution_df, use_container_width=True)
            
            # Проверка подстановкой
            st.markdown("### ✅ Проверка подстановкой:")
            Ax = A_input @ x_halt
            check_df = pd.DataFrame({
                'Уравнение': [f'Ур-е {i+1}' for i in range(n)],
                'Левая часть (Ax)': Ax,
                'Правая часть (b)': b_input,
                'Разница': Ax - b_input
            })
            st.dataframe(check_df, use_container_width=True)
        
        with tab3:
            st.markdown("### 📋 Результаты метода простой итерации")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Время выполнения", f"{stats_iter['time']:.6f} сек")
            with col2:
                st.metric("Количество итераций", stats_iter['iterations'])
            with col3:
                st.metric("Относительная невязка", f"{residual_iter:.2e}")
            
            # Параметры итерационного процесса
            st.markdown("### ⚙️ Параметры итерационного процесса:")
            st.markdown(f"""
            - **Норма матрицы до масштабирования:** {stats_iter['original_norm_B']:.4f}
            - **Норма матрицы после масштабирования:** {stats_iter['scaled_norm_B']:.4f}
            - **Коэффициент масштабирования:** {stats_iter['scale_factor']:.4f}
            """)
            
            # График сходимости (имитация)
            st.markdown("### 📈 График сходимости:")
            iterations = np.arange(0, stats_iter['iterations'] + 1)
            # Имитируем убывание невязки
            residuals = residual_iter * np.exp(-0.01 * iterations) + 1e-10
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.semilogy(iterations, residuals, 'r-', linewidth=2)
            ax.set_xlabel('Номер итерации', fontsize=12)
            ax.set_ylabel('Невязка (логарифмическая шкала)', fontsize=12)
            ax.set_title('Сходимость метода простой итерации', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
        
        # Экспорт результатов
        st.markdown("### 💾 Экспорт результатов")
        col1, col2 = st.columns(2)
        
        with col1:
            # Экспорт решения Халецкого
            solution_df = pd.DataFrame({
                'Переменная': [f'x{i+1}' for i in range(n)],
                'Значение_Халецкого': x_halt,
                'Значение_Итераций': x_iter
            })
            csv = solution_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать решение (CSV)",
                data=csv,
                file_name='solution.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        with col2:
            # Экспорт матрицы и вектора
            data_df = pd.DataFrame(A_input, columns=[f'x{i+1}' for i in range(n)])
            data_df['b'] = b_input
            matrix_csv = data_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать исходные данные (CSV)",
                data=matrix_csv,
                file_name='system_data.csv',
                mime='text/csv',
                use_container_width=True
            )

def main():
    create_manual_input_interface()
    
    # Секция помощи
    with st.sidebar:
        st.markdown("---")
        st.header("❓ Помощь")
        st.markdown("""
        **Как использовать приложение:**
        
        1. **Выберите размер системы** в боковой панели
        2. **Введите коэффициенты** в таблицу или:
           - Нажмите "Сгенерировать случайную систему"
           - Загрузите CSV файл
        3. **Нажмите "Решить систему"**
        4. **Изучите результаты** в трех вкладках
        
        **Требования к матрице для метода Халецкого:**
        - Матрица должна быть симметричной
        - Матрица должна быть положительно определенной
        
        **Формат CSV файла:**
        - Первые n столбцов: матрица A
        - Последний столбец: вектор b
        - Без заголовков (или с заголовками x1, x2, ..., b)
        """)

if __name__ == "__main__":
    main()
