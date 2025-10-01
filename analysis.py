import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

valcode = "EUR"
out_dir = Path(valcode)
out_dir.mkdir(exist_ok=True)

def ts_stats(x: pd.Series) -> dict:
    return {
        "count": int(x.shape[0]),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "var": float(np.var(x, ddof=1)),
        "min": float(np.min(x)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "max": float(np.max(x)),
        "skew": float(stats.skew(x, bias=False)),
        "kurtosis": float(stats.kurtosis(x, fisher=True, bias=False)),
    }

def save_stats_csv(stats_dict: dict, path: Path):
    pd.DataFrame([stats_dict]).to_csv(path, index=False)

# Зчитування даних з файлу
df = pd.read_csv(out_dir / f"data/NBU_{valcode}_UAH.csv", parse_dates=["date"])

y = df["rate"].astype(float).values
t = np.arange(len(y)).reshape(-1, 1)

# Лінійний тренд
lin = LinearRegression().fit(t, y)
y_lin = lin.predict(t)
r2_lin = r2_score(y, y_lin)

# Квадратичний тренд
poly = PolynomialFeatures(degree=2, include_bias=True)
T2 = poly.fit_transform(t)
quad = LinearRegression().fit(T2, y)
y_quad = quad.predict(T2)
r2_quad = r2_score(y, y_quad)

trend_kind = "quadratic" if r2_quad >= r2_lin else "linear"
y_trend = y_quad if trend_kind == "quadratic" else y_lin

# Збереження трендових оцінок
df_trend = df.copy()
df_trend["trend_kind"] = trend_kind
df_trend["y_trend"] = y_trend
df_trend["residual"] = df_trend["rate"] - df_trend["y_trend"]
df_trend.to_csv(out_dir / f"NBU_{valcode}_UAH_with_trend.csv", index=False)

# Статистики реальних даних: рівень, прирости, залишки
rate_stats = ts_stats(df_trend["rate"])
rets = df_trend["rate"].pct_change().dropna()
rets_stats = ts_stats(rets)
resid_stats = ts_stats(df_trend["residual"])

save_stats_csv(rate_stats, out_dir / f"stats_rate_{valcode}.csv")
save_stats_csv(rets_stats, out_dir / f"stats_returns_{valcode}.csv")
save_stats_csv(resid_stats, out_dir / f"stats_residuals_{valcode}.csv")

# Синтез моделі, аналогічної за трендом і статистиками
np.random.seed(42)
n = len(y)

if trend_kind == "quadratic":
    a0, a1, a2 = quad.intercept_, *quad.coef_[1:]
    quad_no_int = LinearRegression(fit_intercept=False).fit(T2, y)
    c0, c1, c2 = quad_no_int.coef_
    y_trend_syn = (c0 + c1 * t.ravel() + c2 * (t.ravel() ** 2))
else:
    b0 = lin.intercept_
    b1 = lin.coef_[0]
    y_trend_syn = b0 + b1 * t.ravel()

# Параметри шуму з реальних залишків
mu_eps = float(np.mean(df_trend["residual"]))
std_eps = float(np.std(df_trend["residual"], ddof=1))

# Синтетичні дані: тренд + нормальний шум
eps = np.random.normal(loc=mu_eps, scale=std_eps, size=n)
y_syn = y_trend_syn + eps

df_syn = pd.DataFrame({
    "t": np.arange(n),
    "y_syn": y_syn,
    "y_trend_syn": y_trend_syn,
    "eps": eps
})
df_syn.to_csv(out_dir / f"synthetic_like_{valcode}.csv", index=False)

# Верифікація: порівняння статистик і трендової відповідності
syn_stats = ts_stats(df_syn["y_syn"])
syn_resid = df_syn["y_syn"] - (quad.predict(T2) if trend_kind=="quadratic" else lin.predict(t))
syn_resid_stats = ts_stats(pd.Series(syn_resid))

save_stats_csv(syn_stats, out_dir / f"stats_synthetic_{valcode}.csv")
save_stats_csv(syn_resid_stats, out_dir / f"stats_synthetic_residuals_{valcode}.csv")

# ====== Графіки ======

# 1. Реальні дані + тренд
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["rate"], label="Справжні дані", color="blue")
plt.plot(df["date"], df_trend["y_trend"], label=f"{trend_kind.capitalize()} trend", color="red")
plt.title(f"{valcode}/UAH Реальні ціна + тренд")
plt.xlabel("Дата")
plt.ylabel("Ціна")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / f"plot_rate_trend_{valcode}.png")
plt.close()

# 2. Залишки (residuals)
plt.figure(figsize=(10, 4))
plt.plot(df["date"], df_trend["residual"], label="Residuals", color="purple")
plt.axhline(0, color="black", linestyle="--")
plt.title(f"Залишки (residuals) {valcode}/UAH trend")
plt.xlabel("Дата")
plt.ylabel("Залишки")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / f"plot_residuals_{valcode}.png")
plt.close()

# 3. Гістограма та щільність розподілу залишків
plt.figure(figsize=(8, 5))
sns.histplot(df_trend["residual"], bins=30, kde=True, color="purple")
plt.title("Розподіл залишків")
plt.xlabel("Залишкова цінність")
plt.ylabel("Частота")
plt.tight_layout()
plt.savefig(out_dir / f"plot_residuals_hist_{valcode}.png")
plt.close()

# 4. Автокореляція залишків (ACF і PACF)
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df_trend["residual"], ax=ax[0], lags=40)
plot_pacf(df_trend["residual"], ax=ax[1], lags=40)
ax[0].set_title("ACF залишків")
ax[1].set_title("PACF залишків")
plt.tight_layout()
plt.savefig(out_dir / f"plot_residuals_acf_pacf_{valcode}.png")
plt.close()

# 5. Синтетичні vs реальні дані
plt.figure(figsize=(10, 5))
plt.plot(df["date"], y, label="Рельні дані", color="blue")
plt.plot(df["date"], y_syn, label="Синтетичні дані", color="orange", alpha=0.7)
plt.title(f"Порівняння реального проти синтетичного {valcode}/UAH")
plt.xlabel("Дата")
plt.ylabel("Ціна")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / f"plot_real_vs_synthetic_{valcode}.png")
plt.close()

# 6. Порівняння розподілів (boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=[y, y_syn], orient="h")
plt.yticks([0, 1], ["Реальні", "Синтетичні"])
plt.title("Порівняння розподілів Boxplot: Реальні та синтетичні дані")
plt.tight_layout()
plt.savefig(out_dir / f"plot_boxplot_real_vs_synthetic_{valcode}.png")
plt.close()

# ====== Вивід статистичних характеристик у консоль ======

ukr_labels = {
    "count": "Кількість спостережень",
    "mean": "Середнє значення",
    "std": "Стандартне відхилення",
    "var": "Дисперсія",
    "min": "Мінімум",
    "q25": "25-й процентиль (Q1)",
    "median": "Медіана",
    "q75": "75-й процентиль (Q3)",
    "max": "Максимум",
    "skew": "Асиметрія",
    "kurtosis": "Ексцес"
}
pd.set_option("display.float_format", "{:.6f}".format)

def print_stats_table(name: str, stats_dict: dict):
    df_stats = pd.DataFrame([stats_dict]).T.reset_index()
    df_stats.columns = ["Метрика", "Значення"]
    df_stats["Метрика"] = df_stats["Метрика"].map(ukr_labels).fillna(df_stats["Метрика"])
    print(f"\n=== {name} ===")
    print(df_stats.to_string(index=False))

print_stats_table("Статистика рівня ряду", rate_stats)
print_stats_table("Статистика приростів (returns)", rets_stats)
print_stats_table("Статистика залишків", resid_stats)
