"""
Лабораторна робота №3: Рекурентне згладжування часових рядів
Реалізація фільтрів Калмана (alfa-beta, alfa-beta-gamma) з адаптацією
Автор: Data Science Engineer
Дата: 2025-10-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Налаштування
VALCODE = "EUR"
INPUT_DIR = Path(VALCODE) / "data"
OUTPUT_DIR = Path(VALCODE) / "lab3_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ЧАСТИНА 1: ВИЯВЛЕННЯ ТА КОМПЕНСАЦІЯ АНОМАЛІЙ
# ============================================================================

def detect_anomalies_iqr(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    """
    Виявлення аномалій методом міжквартильного розмаху (IQR)

    Args:
        data: вхідні дані
        k: коефіцієнт для визначення меж (за замовчуванням 1.5)

    Returns:
        Булева маска аномалій
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return (data < lower_bound) | (data > upper_bound)

def detect_anomalies_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Виявлення аномалій методом Z-score

    Args:
        data: вхідні дані
        threshold: поріг Z-score (за замовчуванням 3.0)

    Returns:
        Булева маска аномалій
    """
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

def compensate_anomalies(data: np.ndarray, anomalies: np.ndarray, method: str = 'interpolate') -> np.ndarray:
    """
    Компенсація аномалій

    Args:
        data: вхідні дані
        anomalies: булева маска аномалій
        method: метод компенсації ('interpolate', 'median', 'mean')

    Returns:
        Дані з компенсованими аномаліями
    """
    corrected = data.copy()

    if method == 'interpolate':
        # Лінійна інтерполяція
        indices = np.arange(len(data))
        valid_mask = ~anomalies
        corrected[anomalies] = np.interp(
            indices[anomalies],
            indices[valid_mask],
            data[valid_mask]
        )
    elif method == 'median':
        # Заміна на медіану вікна
        window = 5
        for i in np.where(anomalies)[0]:
            start = max(0, i - window)
            end = min(len(data), i + window + 1)
            window_data = data[start:end]
            corrected[i] = np.median(window_data[~anomalies[start:end]])
    elif method == 'mean':
        # Заміна на середнє значення
        corrected[anomalies] = np.mean(data[~anomalies])

    return corrected

# ============================================================================
# ЧАСТИНА 2: ALFA-BETA ФІЛЬТР КАЛМАНА
# ============================================================================

class AlphaBetaFilter:
    """
    Alfa-Beta фільтр Калмана для згладжування та прогнозування
    """

    def __init__(self, alpha: float, beta: float, x0: float = 0.0, v0: float = 0.0):
        """
        Args:
            alpha: коефіцієнт згладжування позиції (0 < alpha < 1)
            beta: коефіцієнт згладжування швидкості (0 < beta < 2)
            x0: початкова позиція
            v0: початкова швидкість
        """
        self.alpha = alpha
        self.beta = beta
        self.x = x0
        self.v = v0
        self.dt = 1.0  # часовий крок

    def update(self, z: float) -> Tuple[float, float]:
        """
        Оновлення оцінки на основі нового вимірювання

        Args:
            z: нове вимірювання

        Returns:
            (оцінка позиції, оцінка швидкості)
        """
        # Прогноз
        x_pred = self.x + self.v * self.dt

        # Інновація (похибка прогнозу)
        residual = z - x_pred

        # Корекція
        self.x = x_pred + self.alpha * residual
        self.v = self.v + (self.beta / self.dt) * residual

        return self.x, self.v

    def filter(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Фільтрація всього ряду даних

        Returns:
            (згладжені значення, оцінки швидкості, інновації)
        """
        n = len(data)
        x_filtered = np.zeros(n)
        v_filtered = np.zeros(n)
        innovations = np.zeros(n)

        # Ініціалізація
        self.x = data[0]
        self.v = 0.0

        for i in range(n):
            x_pred = self.x + self.v * self.dt
            innovations[i] = data[i] - x_pred
            self.x, self.v = self.update(data[i])
            x_filtered[i] = self.x
            v_filtered[i] = self.v

        return x_filtered, v_filtered, innovations

# ============================================================================
# ЧАСТИНА 3: ALFA-BETA-GAMMA ФІЛЬТР КАЛМАНА
# ============================================================================

class AlphaBetaGammaFilter:
    """
    Alfa-Beta-Gamma фільтр Калмана (враховує прискорення)
    """

    def __init__(self, alpha: float, beta: float, gamma: float,
                 x0: float = 0.0, v0: float = 0.0, a0: float = 0.0):
        """
        Args:
            alpha: коефіцієнт згладжування позиції
            beta: коефіцієнт згладжування швидкості
            gamma: коефіцієнт згладжування прискорення
            x0: початкова позиція
            v0: початкова швидкість
            a0: початкове прискорення
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.x = x0
        self.v = v0
        self.a = a0
        self.dt = 1.0

    def update(self, z: float) -> Tuple[float, float, float]:
        """
        Оновлення оцінки

        Returns:
            (позиція, швидкість, прискорення)
        """
        # Прогноз
        x_pred = self.x + self.v * self.dt + 0.5 * self.a * self.dt**2
        v_pred = self.v + self.a * self.dt

        # Інновація
        residual = z - x_pred

        # Корекція
        self.x = x_pred + self.alpha * residual
        self.v = v_pred + (self.beta / self.dt) * residual
        self.a = self.a + (self.gamma / (0.5 * self.dt**2)) * residual

        return self.x, self.v, self.a

    def filter(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Фільтрація всього ряду

        Returns:
            (згладжені значення, швидкість, прискорення, інновації)
        """
        n = len(data)
        x_filtered = np.zeros(n)
        v_filtered = np.zeros(n)
        a_filtered = np.zeros(n)
        innovations = np.zeros(n)

        # Ініціалізація
        self.x = data[0]
        self.v = 0.0
        self.a = 0.0

        for i in range(n):
            x_pred = self.x + self.v * self.dt + 0.5 * self.a * self.dt**2
            innovations[i] = data[i] - x_pred
            self.x, self.v, self.a = self.update(data[i])
            x_filtered[i] = self.x
            v_filtered[i] = self.v
            a_filtered[i] = self.a

        return x_filtered, v_filtered, a_filtered, innovations

# ============================================================================
# ЧАСТИНА 4: АДАПТИВНИЙ ФІЛЬТР (подолання розходження)
# ============================================================================

class AdaptiveKalmanFilter:
    """
    Адаптивний фільтр Калмана з автоматичною підстройкою параметрів
    на основі статистики інновацій
    """

    def __init__(self, filter_type: str = 'alpha-beta',
                 window_size: int = 20,
                 innovation_threshold: float = 2.0):
        """
        Args:
            filter_type: тип базового фільтра ('alpha-beta' або 'alpha-beta-gamma')
            window_size: розмір вікна для аналізу інновацій
            innovation_threshold: поріг для виявлення розходження
        """
        self.filter_type = filter_type
        self.window_size = window_size
        self.innovation_threshold = innovation_threshold
        self.innovation_history = []

        # Початкові параметри
        if filter_type == 'alpha-beta':
            self.alpha = 0.3
            self.beta = 0.1
            self.filter = None
        else:  # alpha-beta-gamma
            self.alpha = 0.5
            self.beta = 0.3
            self.gamma = 0.1
            self.filter = None

    def adapt_parameters(self, innovation: float):
        """
        Адаптація параметрів фільтра на основі інновації
        """
        self.innovation_history.append(innovation)

        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)

        if len(self.innovation_history) >= self.window_size:
            # Обчислення статистики інновацій
            innovations = np.array(self.innovation_history)
            innovation_std = np.std(innovations)
            innovation_mean = np.abs(np.mean(innovations))

            # Виявлення розходження
            if innovation_std > self.innovation_threshold * np.std(innovations[:self.window_size//2]):
                # Збільшуємо коефіцієнти для швидшої адаптації
                if self.filter_type == 'alpha-beta':
                    self.alpha = min(0.9, self.alpha * 1.2)
                    self.beta = min(0.5, self.beta * 1.2)
                else:
                    self.alpha = min(0.9, self.alpha * 1.2)
                    self.beta = min(0.5, self.beta * 1.2)
                    self.gamma = min(0.3, self.gamma * 1.2)
            else:
                # Зменшуємо коефіцієнти для більшого згладжування
                if self.filter_type == 'alpha-beta':
                    self.alpha = max(0.05, self.alpha * 0.95)
                    self.beta = max(0.01, self.beta * 0.95)
                else:
                    self.alpha = max(0.1, self.alpha * 0.95)
                    self.beta = max(0.05, self.beta * 0.95)
                    self.gamma = max(0.01, self.gamma * 0.95)

    def filter_data(self, data: np.ndarray) -> Tuple[np.ndarray, List[float], List[float]]:
        """
        Фільтрація з адаптацією

        Returns:
            (згладжені дані, історія alpha, інновації)
        """
        n = len(data)
        x_filtered = np.zeros(n)
        alpha_history = []
        innovations_out = []

        if self.filter_type == 'alpha-beta':
            flt = AlphaBetaFilter(self.alpha, self.beta, data[0], 0.0)
        else:
            flt = AlphaBetaGammaFilter(self.alpha, self.beta, self.gamma, data[0], 0.0, 0.0)

        for i in range(n):
            # Прогноз
            if self.filter_type == 'alpha-beta':
                x_pred = flt.x + flt.v * flt.dt
            else:
                x_pred = flt.x + flt.v * flt.dt + 0.5 * flt.a * flt.dt**2

            # Інновація
            innovation = data[i] - x_pred
            innovations_out.append(innovation)

            # Адаптація параметрів
            self.adapt_parameters(innovation)

            # Оновлення параметрів фільтра
            flt.alpha = self.alpha
            flt.beta = self.beta
            if self.filter_type == 'alpha-beta-gamma':
                flt.gamma = self.gamma

            # Оновлення стану
            if self.filter_type == 'alpha-beta':
                x_filtered[i], _ = flt.update(data[i])
            else:
                x_filtered[i], _, _ = flt.update(data[i])

            alpha_history.append(self.alpha)

        return x_filtered, alpha_history, innovations_out

# ============================================================================
# ЧАСТИНА 5: ОЦІНКА ЯКОСТІ ФІЛЬТРАЦІЇ
# ============================================================================

def calculate_metrics(original: np.ndarray, filtered: np.ndarray, name: str = "") -> Dict:
    """
    Обчислення метрик якості фільтрації
    """
    residuals = original - filtered

    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))

    # Перевірка незміщеності
    bias = np.mean(residuals)

    # SNR (Signal-to-Noise Ratio)
    signal_power = np.var(filtered)
    noise_power = np.var(residuals)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    return {
        'filter_name': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'bias': bias,
        'SNR_dB': snr,
        'std_residuals': np.std(residuals),
        'is_unbiased': np.abs(bias) < 0.01 * np.std(original)
    }

# ============================================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================================

def main():
    print("="*80)
    print("ЛАБОРАТОРНА РОБОТА №3: Рекурентне згладжування фільтрами Калмана")
    print("="*80)

    # 1. Завантаження даних
    print("\n[1] Завантаження даних...")
    df = pd.read_csv(INPUT_DIR / f"NBU_{VALCODE}_UAH.csv", parse_dates=['date'])
    data_original = df['rate'].values
    dates = df['date'].values
    n = len(data_original)
    print(f"Завантажено {n} спостережень")

    # 2. Виявлення аномалій
    print("\n[2] Виявлення аномалій...")
    anomalies_iqr = detect_anomalies_iqr(data_original, k=1.5)
    anomalies_zscore = detect_anomalies_zscore(data_original, threshold=3.0)
    anomalies_combined = anomalies_iqr | anomalies_zscore

    n_anomalies = np.sum(anomalies_combined)
    print(f"Виявлено {n_anomalies} аномалій ({100*n_anomalies/n:.2f}%)")

    # 3. Компенсація аномалій
    print("\n[3] Компенсація аномалій...")
    data_corrected = compensate_anomalies(data_original, anomalies_combined, method='interpolate')

    # Збереження даних з аномаліями
    df_anomalies = df.copy()
    df_anomalies['is_anomaly'] = anomalies_combined
    df_anomalies['rate_corrected'] = data_corrected
    df_anomalies.to_csv(OUTPUT_DIR / 'anomalies_detected.csv', index=False)

    # 4. Alfa-Beta фільтр
    print("\n[4] Застосування Alfa-Beta фільтра...")
    ab_filter = AlphaBetaFilter(alpha=0.3, beta=0.1, x0=data_corrected[0])
    ab_filtered, ab_velocity, ab_innovations = ab_filter.filter(data_corrected)

    metrics_ab = calculate_metrics(data_corrected, ab_filtered, "Alfa-Beta")
    print(f"  RMSE: {metrics_ab['RMSE']:.4f}")
    print(f"  Bias: {metrics_ab['bias']:.6f}")
    print(f"  Незміщений: {'ТАК' if metrics_ab['is_unbiased'] else 'НІ'}")

    # 5. Alfa-Beta-Gamma фільтр
    print("\n[5] Застосування Alfa-Beta-Gamma фільтра...")
    abg_filter = AlphaBetaGammaFilter(alpha=0.5, beta=0.3, gamma=0.1, x0=data_corrected[0])
    abg_filtered, abg_velocity, abg_acceleration, abg_innovations = abg_filter.filter(data_corrected)

    metrics_abg = calculate_metrics(data_corrected, abg_filtered, "Alfa-Beta-Gamma")
    print(f"  RMSE: {metrics_abg['RMSE']:.4f}")
    print(f"  Bias: {metrics_abg['bias']:.6f}")
    print(f"  Незміщений: {'ТАК' if metrics_abg['is_unbiased'] else 'НІ'}")

    # 6. Адаптивний фільтр
    print("\n[6] Застосування Адаптивного фільтра...")
    adaptive_filter = AdaptiveKalmanFilter(filter_type='alpha-beta-gamma', window_size=20)
    adaptive_filtered, alpha_history, adaptive_innovations = adaptive_filter.filter_data(data_corrected)

    metrics_adaptive = calculate_metrics(data_corrected, adaptive_filtered, "Adaptive Kalman")
    print(f"  RMSE: {metrics_adaptive['RMSE']:.4f}")
    print(f"  Bias: {metrics_adaptive['bias']:.6f}")
    print(f"  Незміщений: {'ТАК' if metrics_adaptive['is_unbiased'] else 'НІ'}")

    # 7. Збереження результатів
    print("\n[7] Збереження результатів...")

    # Результати фільтрації
    df_results = pd.DataFrame({
        'date': dates,
        'original': data_original,
        'corrected': data_corrected,
        'alpha_beta': ab_filtered,
        'alpha_beta_gamma': abg_filtered,
        'adaptive': adaptive_filtered,
        'is_anomaly': anomalies_combined
    })
    df_results.to_csv(OUTPUT_DIR / 'filtered_results.csv', index=False)

    # Метрики
    df_metrics = pd.DataFrame([metrics_ab, metrics_abg, metrics_adaptive])
    df_metrics.to_csv(OUTPUT_DIR / 'filter_metrics.csv', index=False)

    # Інновації
    df_innovations = pd.DataFrame({
        'date': dates,
        'alpha_beta': ab_innovations,
        'alpha_beta_gamma': abg_innovations,
        'adaptive': adaptive_innovations
    })
    df_innovations.to_csv(OUTPUT_DIR / 'innovations.csv', index=False)

    # Адаптивні параметри
    df_adaptive_params = pd.DataFrame({
        'date': dates,
        'alpha': alpha_history
    })
    df_adaptive_params.to_csv(OUTPUT_DIR / 'adaptive_parameters.csv', index=False)

    # 8. Візуалізація
    print("\n[8] Створення візуалізацій...")

    # График 1: Порівняння всіх фільтрів
    plt.figure(figsize=(14, 6))
    plt.plot(dates, data_corrected, 'o-', label='Скориговані дані', alpha=0.3, markersize=2)
    plt.plot(dates, ab_filtered, '-', label='Alfa-Beta', linewidth=2)
    plt.plot(dates, abg_filtered, '-', label='Alfa-Beta-Gamma', linewidth=2)
    plt.plot(dates, adaptive_filtered, '-', label='Adaptive', linewidth=2)
    plt.title(f'Порівняння фільтрів Калмана - {VALCODE}/UAH', fontsize=14, fontweight='bold')
    plt.xlabel('Дата')
    plt.ylabel('Курс')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_all_filters.png', dpi=300)
    plt.close()

    # График 2: Аномалії
    plt.figure(figsize=(14, 6))
    plt.plot(dates, data_original, 'o-', label='Оригінальні дані', alpha=0.5, markersize=3)
    plt.scatter(dates[anomalies_combined], data_original[anomalies_combined],
                color='red', s=100, marker='x', label=f'Аномалії (n={n_anomalies})', zorder=5)
    plt.plot(dates, data_corrected, '-', label='Скориговані дані', linewidth=2, color='green')
    plt.title('Виявлення та компенсація аномалій', fontsize=14, fontweight='bold')
    plt.xlabel('Дата')
    plt.ylabel('Курс')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'anomalies_detection.png', dpi=300)
    plt.close()

    # График 3: Інновації (residuals)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(dates, ab_innovations, 'o-', markersize=2, alpha=0.6)
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('Інновації: Alfa-Beta фільтр')
    axes[0].set_ylabel('Інновація')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, abg_innovations, 'o-', markersize=2, alpha=0.6, color='orange')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Інновації: Alfa-Beta-Gamma фільтр')
    axes[1].set_ylabel('Інновація')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dates, adaptive_innovations, 'o-', markersize=2, alpha=0.6, color='green')
    axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title('Інновації: Адаптивний фільтр')
    axes[2].set_ylabel('Інновація')
    axes[2].set_xlabel('Дата')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'innovations_comparison.png', dpi=300)
    plt.close()

    # График 4: Адаптація параметра alpha
    plt.figure(figsize=(14, 6))
    plt.plot(dates, alpha_history, '-', linewidth=2)
    plt.title('Адаптація параметра α (alpha) у часі', fontsize=14, fontweight='bold')
    plt.xlabel('Дата')
    plt.ylabel('Значення α')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'adaptive_alpha_evolution.png', dpi=300)
    plt.close()

    # График 5: Гістограма інновацій
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(ab_innovations, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title('Розподіл інновацій: Alfa-Beta')
    axes[0].set_xlabel('Інновація')
    axes[0].set_ylabel('Частота')
    axes[0].axvline(0, color='red', linestyle='--')

    axes[1].hist(abg_innovations, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_title('Розподіл інновацій: Alfa-Beta-Gamma')
    axes[1].set_xlabel('Інновація')
    axes[1].axvline(0, color='red', linestyle='--')

    axes[2].hist(adaptive_innovations, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[2].set_title('Розподіл інновацій: Adaptive')
    axes[2].set_xlabel('Інновація')
    axes[2].axvline(0, color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'innovations_distribution.png', dpi=300)
    plt.close()

    # График 6: Похибки фільтрації
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    residuals_ab = data_corrected - ab_filtered
    residuals_abg = data_corrected - abg_filtered
    residuals_adaptive = data_corrected - adaptive_filtered

    axes[0].plot(dates, residuals_ab, 'o-', markersize=2, alpha=0.6)
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Похибки: Alfa-Beta (RMSE={metrics_ab["RMSE"]:.4f})')
    axes[0].set_ylabel('Похибка')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, residuals_abg, 'o-', markersize=2, alpha=0.6, color='orange')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(f'Похибки: Alfa-Beta-Gamma (RMSE={metrics_abg["RMSE"]:.4f})')
    axes[1].set_ylabel('Похибка')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dates, residuals_adaptive, 'o-', markersize=2, alpha=0.6, color='green')
    axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title(f'Похибки: Adaptive (RMSE={metrics_adaptive["RMSE"]:.4f})')
    axes[2].set_ylabel('Похибка')
    axes[2].set_xlabel('Дата')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'filtering_errors.png', dpi=300)
    plt.close()

    print("\n" + "="*80)
    print("ЗАВЕРШЕНО!")
    print("="*80)
    print(f"\nВсі результати збережено в: {OUTPUT_DIR}/")
    print("\nФайли:")
    print("  - filtered_results.csv - результати фільтрації")
    print("  - filter_metrics.csv - метрики якості")
    print("  - innovations.csv - інновації фільтрів")
    print("  - adaptive_parameters.csv - еволюція параметрів")
    print("  - anomalies_detected.csv - виявлені аномалії")
    print("  - 6 PNG графіків")

    print("\n" + "="*80)
    print("ВИСНОВКИ:")
    print("="*80)

    # Вибір найкращого фільтра
    best_filter = min([metrics_ab, metrics_abg, metrics_adaptive], key=lambda x: x['RMSE'])
    print(f"\nНайкращий фільтр за RMSE: {best_filter['filter_name']}")
    print(f"  RMSE: {best_filter['RMSE']:.6f}")
    print(f"  MAE: {best_filter['MAE']:.6f}")
    print(f"  Bias: {best_filter['bias']:.6f}")
    print(f"  SNR: {best_filter['SNR_dB']:.2f} dB")
    print(f"  Незміщеність: {'ТАК ✓' if best_filter['is_unbiased'] else 'НІ ✗'}")

    print("\nОбґрунтування вибору фільтра:")
    if best_filter['filter_name'] == 'Alfa-Beta-Gamma':
        print("  Alfa-Beta-Gamma фільтр враховує прискорення (другу похідну),")
        print("  що дозволяє краще відслідковувати нелінійні тренди у курсі EUR/UAH.")
    elif best_filter['filter_name'] == 'Adaptive Kalman':
        print("  Адаптивний фільтр автоматично підлаштовує параметри під динаміку ряду,")
        print("  що дозволяє уникнути розходження та забезпечує стабільність оцінок.")
    else:
        print("  Alfa-Beta фільтр забезпечує баланс між простотою та якістю,")
        print("  підходить для рядів з помірною нелінійністю.")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()