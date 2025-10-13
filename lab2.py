import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import datetime
import os

# ----------------- Налаштування -----------------
VALCODE = "EUR"
BASE_DIR = Path(VALCODE)
DATA_PATH = BASE_DIR / "data" / f"NBU_{VALCODE}_UAH.csv"
OUT_DIR = BASE_DIR / "lab2_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DPI = 150
np.random.seed(42)

# ----------------- Допоміжні функції -----------------
def ts_stats(x: pd.Series) -> dict:
    import scipy.stats as stats
    arr = np.asarray(x.dropna(), dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "var": float(np.var(arr, ddof=1)),
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
        "skew": float(stats.skew(arr, bias=False)),
        "kurtosis": float(stats.kurtosis(arr, fisher=True, bias=False)),
    }

def save_stats_csv(stats_dict: dict, path: Path):
    pd.DataFrame([stats_dict]).to_csv(path, index=False)

def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

# Реалізація МНК (LSM) через розв'язання нормальних рівнянь/псевдообернену (np.linalg.lstsq)
def fit_polynomial_lsm(t, y, degree):
    # t: 1D array of time indices (n,)
    # y: 1D array of targets (n,)
    # повертаємо коефіцієнти у вигляді [a0, a1, a2, ...] де y ≈ a0 + a1*t + a2*t^2 + ...
    X = np.vander(t, N=degree+1, increasing=True)  # shape (n, degree+1)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs  # length degree+1

def predict_polynomial(coeffs, t):
    t = np.asarray(t, dtype=float)
    y_pred = np.zeros_like(t, dtype=float)
    for p, c in enumerate(coeffs):
        y_pred += c * np.power(t, p, where=np.isfinite(t))
    return y_pred

# ----------------- 1) Зчитування даних -----------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Не знайдено дані за шляхом {DATA_PATH}. Запустіть parser_lab1.py раніше.")

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df["rate"] = df["rate"].astype(float)

n = len(df)
t = np.arange(n)

# ----------------- 2) Очищення даних від аномалій -----------------
# Комбінований підхід: IQR + Z-score.
series = df["rate"].copy()

# IQR метод
Q1 = series.quantile(0.25)
Q3 = series.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
mask_iqr = (series < lower) | (series > upper)

# Z-score метод (global)
mu = series.mean()
sigma = series.std(ddof=1)
mask_z = np.abs((series - mu) / sigma) > 3.0

# Об'єднання масок
mask_outliers = mask_iqr | mask_z
outlier_indices = np.where(mask_outliers)[0]

df_clean = df.copy()
df_clean["is_outlier"] = False
df_clean.loc[outlier_indices, "is_outlier"] = True

# Очищення: лінійна інтерполяція через пропуски
df_clean.loc[outlier_indices, "rate"] = np.nan
df_clean["rate"] = df_clean["rate"].interpolate(method="linear", limit_direction="both")
# Якщо початкові/кінцеві NaN залишилися, заповнимо forward/backward
df_clean["rate"] = df_clean["rate"].ffill().bfill()

# Збереження інформації
df_clean.to_csv(OUT_DIR / "data_cleaned.csv", index=False)
print(f"Found {len(outlier_indices)} outliers; saved cleaned data to {OUT_DIR/'data_cleaned.csv'}")

# ----------------- 3) Підготовка тренувальної / валідаційної частини (TimeSeriesSplit по часу) -----------------
y = df_clean["rate"].values
t_indices = np.arange(len(y))

val_size = max(1, int(0.2 * len(y)))
train_size = len(y) - val_size
t_train = t_indices[:train_size]
y_train = y[:train_size]
t_val = t_indices[train_size:]
y_val = y[train_size:]

# ----------------- 4) Визначення показника якості і оптимізація моделі -----------------
# Показник якості: RMSE на валідації.
degrees = list(range(1, 3))
results = []

for deg in degrees:
    coeffs = fit_polynomial_lsm(t_train, y_train, deg)
    y_val_pred = predict_polynomial(coeffs, t_val)
    score_rmse = rmse(y_val, y_val_pred)
    score_mae = mean_absolute_error(y_val, y_val_pred)
    score_r2 = r2_score(y_val, y_val_pred)
    results.append({
        "degree": deg,
        "rmse": float(score_rmse),
        "mae": float(score_mae),
        "r2": float(score_r2),
        "coeffs": coeffs
    })
pd.DataFrame([{k: v for k, v in r.items() if k != "coeffs"} for r in results]).to_csv(OUT_DIR / "model_selection.csv", index=False)

# Вибір найкращого за RMSE
best = min(results, key=lambda x: x["rmse"])
best_degree = best["degree"]
best_coeffs = best["coeffs"]
print(f"Best degree by RMSE on validation: {best_degree} (RMSE={best['rmse']:.6f}, R2={best['r2']:.6f})")

# ----------------- 5) Навчання фінальної моделі на ВСІХ даних (щоб екстраполювати) -----------------
final_coeffs = fit_polynomial_lsm(t_indices, y, best_degree)
y_fit_all = predict_polynomial(final_coeffs, t_indices)
residuals = y - y_fit_all
residuals = np.where(np.isfinite(residuals), residuals, np.nan)

rate_stats = ts_stats(pd.Series(y))
resid_stats = ts_stats(pd.Series(residuals).dropna())
save_stats_csv(rate_stats, OUT_DIR / "stats_rate.csv")
save_stats_csv(resid_stats, OUT_DIR / "stats_residuals.csv")

# ----------------- 6) Прогноз (екстраполяція) на 0.5 інтервалу вибірки -----------------
h = int(0.5 * len(y))
t_forecast = np.arange(len(y), len(y) + h)
y_forecast = predict_polynomial(final_coeffs, t_forecast)

df_forecast = pd.DataFrame({
    "date": [df_clean["date"].iloc[-1] + datetime.timedelta(days=i+1) for i in range(h)],
    "t": t_forecast,
    "forecast": y_forecast
})
df_forecast.to_csv(OUT_DIR / "forecast.csv", index=False)

# ----------------- 7) Верифікація: статистики прогнозу та порівняння -----------------
syn_resid_stats = ts_stats(pd.Series(residuals))
save_stats_csv(syn_resid_stats, OUT_DIR / "stats_model_residuals.csv")

# ----------------- 8) Збереження даних з трендом/залишками -----------------
df_out = df_clean.copy()
df_out["y_fit"] = y_fit_all
df_out["residual"] = residuals
df_out.to_csv(OUT_DIR / "data_with_trend_and_residuals.csv", index=False)

# ----------------- 9) Графіки -----------------
# 9.1 Реальні дані + підгонка
plt.figure(figsize=(10,5), dpi=PLOT_DPI)
plt.plot(df_out["date"], df_out["rate"], label="Реальні дані", linewidth=1)
plt.plot(df_out["date"], df_out["y_fit"], label=f"Поліном (deg={best_degree})", linewidth=1.5)
plt.plot(df_forecast["date"], df_forecast["forecast"], label="Прогноз (екстраполяція)", linestyle="--")
plt.title(f"{VALCODE}/UAH: Дані + Підгонка + Прогноз")
plt.xlabel("Дата")
plt.ylabel("Курс")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_fit_and_forecast.png")
plt.close()

# 9.2 Залишки
plt.figure(figsize=(10,4), dpi=PLOT_DPI)
plt.plot(df_out["date"], df_out["residual"], label="Залишки")
plt.axhline(0, linestyle="--", color="k")
plt.title("Залишки моделі")
plt.xlabel("Дата")
plt.ylabel("Залишки")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_residuals.png")
plt.close()

# 9.3 Гістограма залишків
resid_clean = df_out["residual"].replace([np.inf, -np.inf], np.nan).dropna()
plt.figure(figsize=(8,5), dpi=PLOT_DPI)
plt.hist(resid_clean, bins=30, density=True, alpha=0.7)
plt.title("Гістограма залишків")
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_residuals_hist.png")
plt.close()

# 9.4 Диаграма: тренувальна / валідаційна похибка за степенем (з результатів)
sel = pd.DataFrame(results)
plt.figure(figsize=(8,5), dpi=PLOT_DPI)
plt.plot(sel["degree"], sel["rmse"], marker="o")
plt.title("RMSE на валідації vs степінь полінома")
plt.xlabel("Степінь полінома")
plt.ylabel("RMSE (validation)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_rmse_by_degree.png")
plt.close()

# ----------------- 10) Збереження метрик та вивід -----------------
pd.DataFrame(results).to_csv(OUT_DIR / "model_candidates_full.csv", index=False)

summary = {
    "best_degree": best_degree,
    "best_rmse_val": float(best["rmse"]),
    "best_r2_val": float(best["r2"]),
    "train_size": int(train_size),
    "val_size": int(val_size),
    "forecast_h": int(h)
}
pd.DataFrame([summary]).to_csv(OUT_DIR / "summary.csv", index=False)

print("\n==== РЕЗУЛЬТАТИ ====")
print(f"Кращий степінь полінома (за RMSE): {best_degree}")
print(f"RMSE (валідація): {best['rmse']:.6f}, R2 (валідація): {best['r2']:.6f}")
print(f"Розмір вибірки: {len(y)}, Прогноз на h={h} точок (0.5 * n)")
print(f"Saved outputs to: {OUT_DIR.resolve()}")

# Вивід основних статистик в консоль
pd.options.display.float_format = "{:.6f}".format
print("\n--- Статистика рівня ряду ---")
print(pd.DataFrame([rate_stats]).T)
print("\n--- Статистика залишків моделі ---")
print(pd.DataFrame([resid_stats]).T)