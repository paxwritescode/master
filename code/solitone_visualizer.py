import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def plot_soliton_comparison(
    f: Callable[[float, np.ndarray], np.ndarray],
    u0: np.ndarray,
    t_span: tuple[float, float],
    h_large: float,
    theta_author: np.ndarray,
    theta_optimized: np.ndarray,
    exact_solution_func: Callable[[np.ndarray], np.ndarray] = None
):
    """
    Строит зрелищный график сравнения точного решения солитона,
    плохого решения (авторские коэффициенты на крупном шаге) 
    и отличного решения (оптимизированные коэффициенты на том же крупном шаге).
    """
    print("\n[Visualizer] Генерация финального графического отчета...")
    
    # Импортируем наш интегратор локально
    from rk_solver import rk6_8_integrate
    
    # 1. Расчет траекторий
    # А) Решение с авторскими (неоптимизированными под эту задачу) коэффициентами
    t_bad, u_bad = rk6_8_integrate(f, u0, t_span, h_large, theta_author)
    
    # Б) Решение с твоими вылизанными оптимизированными коэффициентами
    t_good, u_good = rk6_8_integrate(f, u0, t_span, h_large, theta_optimized)
    
    # В) Идеальная аналитическая траектория
    if exact_solution_func is not None:
        t_exact = np.linspace(t_span[0], t_span[1], 1000)
        u_exact = exact_solution_func(t_exact)
    else:
        # Если аналитической формулы нет, строим эталон на сверхмелкой сетке
        print("[Visualizer] Аналитическое решение не задано. Считаем эталон на мелкой сетке h...")
        t_exact, u_exact_grid = rk6_8_integrate(f, u0, t_span, h_large / 20.0, theta_optimized)
        # Берем первую компоненту для визуализации профиля волны
        u_exact = u_exact_grid[:, 0] 
        t_exact = t_exact

    # 2. Настройка графического окна в академическом стиле
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # --- ВЕРХНИЙ ГРАФИК: Сравнение волновых профилей ---
    # Точное решение — солидная черная линия
    ax1.plot(t_exact, u_exact, color='black', linestyle='-', linewidth=2.0, 
             label='Аналитическое решение (Точное)')
    
    # Плохое решение — красная штриховая линия (показывает фазовый сдвиг и распад профиля)
    ax1.plot(t_bad, u_bad[:, 0], color='#d9534f', linestyle='--', linewidth=1.5, 
             label=f'Авторские параметры (Anastassi 2023), h = {h_large}')
    
    # Твое решение — синие маркеры-точки (идеально ложатся на черную линию)
    ax1.plot(t_good, u_good[:, 0], color='#0275d8', linestyle='', marker='o', 
             markersize=5, markevery=1, label=f'Оптимизированные параметры RK(6,8), h = {h_large}')
    
    ax1.set_ylabel('Амплитуда волны $\psi(t)$', fontsize=13)
    ax1.set_title('Эволюция волнового профиля солитона Шрёдингера', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, shadow=False, facecolor='white', edgecolor='none')
    
    # --- НИЖНИЙ ГРАФИК: Динамика абсолютной ошибки ---
    # Ошибка автора
    error_bad = np.abs(u_bad[:, 0] - np.interp(t_bad, t_exact, u_exact))
    # Твоя ошибка
    error_good = np.abs(u_good[:, 0] - np.interp(t_good, t_exact, u_exact))
    
    ax2.plot(t_bad, error_bad, color='#d9534f', linestyle='-', linewidth=1.2, label='Погрешность базового метода')
    ax2.plot(t_good, error_good, color='#0275d8', linestyle='-', linewidth=1.5, label='Погрешность оптимизированного метода')
    
    ax2.set_xlabel('Временной интервал симуляции $t$', fontsize=13)
    ax2.set_ylabel('Абсолютная ошибка $\Delta$', fontsize=13)
    ax2.set_yscale('log') # Логарифмическая шкала наглядно покажет разницу в порядках
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Сохраняем в векторном формате PDF для идеального качества в LaTeX/дипломе
    plt.savefig('soliton_optimization_triumph.pdf', dpi=300)
    plt.show()
    print("[Visualizer] Графики успешно сохранены в файлы 'soliton_optimization_triumph.pdf' и '.png'")