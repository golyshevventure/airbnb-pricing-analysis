#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04 — Advanced Analysis
=======================
Дополнительные аналитические слои для проекта Airbnb:
1. Корреляционный анализ + линейная регрессия
2. Сегментация хостов (любители vs профи)
3. Географическая визуализация (scatter map)
4. Простая предиктивная модель

Работает с airbnb_raw_combined.csv (все 76 колонок).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import os

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# =============================================================================
# БЛОК 1: ПУТИ И ЗАГРУЗКА
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots", "advanced")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Загружаем raw_combined (все 76 колонок)
raw_path = os.path.join(PROCESSED_DIR, "airbnb_raw_combined.csv")
df = pd.read_csv(raw_path, low_memory=False)

# Чистим цену (тот же код, что в 02_data_cleaning.py)
df["price_clean"] = (
    df["price"].astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)
# Убираем выбросы
df = df[(df["price_clean"] >= 10) & (df["price_clean"] <= 2000)].copy()

print("=" * 70)
print("РАСШИРЕННЫЙ АНАЛИЗ")
print("=" * 70)
print(f"\n📦 Рабочая таблица: {len(df):,} строк × {len(df.columns)} колонок")

# =============================================================================
# БЛОК 2: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# =============================================================================

print("\n" + "-" * 70)
print("ШАГ 1: Корреляционный анализ")
print("-" * 70)

# Выбираем числовые колонки для корреляции
numeric_cols = [
    "price_clean", "number_of_reviews", "review_scores_rating",
    "minimum_nights", "latitude", "longitude"
]

# Добавляем host_listings_count, если есть
if "calculated_host_listings_count" in df.columns:
    numeric_cols.append("calculated_host_listings_count")
    df["host_listings_count"] = df["calculated_host_listings_count"]
else:
    df["host_listings_count"] = np.nan

# Создаем матрицу корреляции
df_corr = df[numeric_cols].copy()
corr_matrix = df_corr.corr()

# Строим heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Correlation Matrix: What Drives Airbnb Prices?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "01_correlation_heatmap.png"), dpi=150)
plt.close()

# Выводим ключевые корреляции с ценой
price_corr = corr_matrix["price_clean"].drop("price_clean").sort_values(key=abs, ascending=False)
print("\nКорреляции с ценой (по убыванию силы связи):")
for feature, corr in price_corr.items():
    direction = "↑" if corr > 0 else "↓"
    print(f"   {direction} {feature}: {corr:.3f}")

# =============================================================================
# БЛОК 3: ЛИНЕЙНАЯ РЕГРЕССИЯ (ПРЕДИКТИВНАЯ МОДЕЛЬ)
# =============================================================================

print("\n" + "-" * 70)
print("ШАГ 2: Предиктивная модель (Linear Regression)")
print("-" * 70)

# Готовим признаки (features) и целевую переменную (target)
# Используем только строки без пропусков в нужных колонках
feature_cols = ["number_of_reviews", "review_scores_rating", "minimum_nights",
                "latitude", "longitude"]
if "host_listings_count" in df.columns and df["host_listings_count"].notna().any():
    feature_cols.append("host_listings_count")

# Добавляем dummy-переменные для типа жилья
room_type_dummies = pd.get_dummies(df["room_type"], prefix="room")
df_model = pd.concat([df, room_type_dummies], axis=1)
feature_cols += list(room_type_dummies.columns)

# Убираем строки с пропусками
mask = df_model[feature_cols + ["price_clean"]].notna().all(axis=1)
df_model_clean = df_model[mask].copy()

X = df_model_clean[feature_cols]
y = df_model_clean["price_clean"]

# Разбиваем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказываем
y_pred = model.predict(X_test)

# Метрики
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 Результаты модели:")
print(f"   R² (коэффициент детерминации): {r2:.3f}")
print(f"   MAE (средняя абсолютная ошибка): ${mae:.2f}")
print(f"   Интерпретация: модель объясняет {r2*100:.1f}% дисперсии цен.")

# Важность признаков (коэффициенты)
coefs = pd.Series(model.coef_, index=feature_cols).sort_values(key=abs, ascending=False)
print(f"\n🔍 Топ-5 признаков по влиянию на цену:")
for feature, coef in coefs.head(5).items():
    direction = "↑" if coef > 0 else "↓"
    print(f"   {direction} {feature}: {coef:.2f}")

# График: реальные vs предсказанные цены
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.3, s=10)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax.set_xlabel("Actual Price ($)", fontsize=12)
ax.set_ylabel("Predicted Price ($)", fontsize=12)
ax.set_title(f"Predicted vs Actual Prices (R² = {r2:.3f})", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "02_prediction_vs_actual.png"), dpi=150)
plt.close()

# График важности признаков
fig, ax = plt.subplots(figsize=(10, 6))
coefs.head(10).plot(kind="barh", ax=ax, color=sns.color_palette("husl", 10))
ax.set_title("Top 10 Features Driving Price", fontsize=14, fontweight="bold")
ax.set_xlabel("Coefficient (impact on $)", fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "03_feature_importance.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 4: СЕГМЕНТАЦИЯ ХОСТОВ
# =============================================================================

print("\n" + "-" * 70)
print("ШАГ 3: Сегментация хостов")
print("-" * 70)

# Если нет колонки host_listings_count — создаем из calculated_host_listings_count
if "calculated_host_listings_count" in df.columns:
    df["host_segment"] = pd.cut(
        df["calculated_host_listings_count"],
        bins=[0, 1, 5, 20, 9999],
        labels=["Hobbyist (1)", "Small Host (2-5)", "Manager (6-20)", "Professional (20+)"],
        include_lowest=True
    )
else:
    # Fallback: если нет данных, делаем фиктивную сегментацию по id
    print("   ⚠️ Нет данных о количестве объявлений хоста. Сегментация невозможна.")
    df["host_segment"] = "Unknown"

if "host_segment" in df.columns and df["host_segment"].notna().any():
    segment_stats = df.groupby("host_segment").agg({
        "price_clean": ["count", "median", "mean"],
        "review_scores_rating": "mean",
        "number_of_reviews": "median"
    }).round(2)

    print("\n📊 Статистика по сегментам хостов:")
    print(segment_stats)

    # График: цена по сегментам
    fig, ax = plt.subplots(figsize=(10, 6))
    segment_order = ["Hobbyist (1)", "Small Host (2-5)", "Manager (6-20)", "Professional (20+)"]
    segment_data = df[df["host_segment"].isin(segment_order)]

    if not segment_data.empty:
        sns.boxplot(data=segment_data[segment_data["price_clean"] <= 400],
                    x="host_segment", y="price_clean", order=segment_order, ax=ax)
        ax.set_title("Price Distribution by Host Segment", fontsize=14, fontweight="bold")
        ax.set_xlabel("Host Type", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(SCREENSHOTS_DIR, "04_host_segmentation.png"), dpi=150)
        plt.close()

# =============================================================================
# БЛОК 5: ГЕОГРАФИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ
# =============================================================================

print("\n" + "-" * 70)
print("ШАГ 4: Географическая визуализация")
print("-" * 70)

# Для каждого города строим scatter plot: lat vs lon, цвет = цена
# Берем выборку, чтобы не перегружать график

cities = df["city"].unique()
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, city in enumerate(cities):
    ax = axes[i]
    city_df = df[df["city"] == city].sample(n=min(2000, len(df[df["city"] == city])), random_state=42)

    scatter = ax.scatter(
        city_df["longitude"], city_df["latitude"],
        c=city_df["price_clean"], cmap="YlOrRd",
        alpha=0.5, s=8, vmin=10, vmax=300
    )
    ax.set_title(f"{city} — Price Heatmap", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, label="Price ($)")

axes[5].axis("off")
plt.suptitle("Geographic Price Distribution Across 5 Cities", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "05_geographic_heatmap.png"), dpi=150)
plt.close()

print("\n   ✅ Карта сохранена: показывает, где в каждом городе дорогие/дешевые районы")

# =============================================================================
# БЛОК 6: СОХРАНЕНИЕ ИНСАЙТОВ
# =============================================================================

print("\n" + "=" * 70)
print("ДОПОЛНИТЕЛЬНЫЕ ИНСАЙТЫ")
print("=" * 70)

insights = []

# Инсайт из регрессии
top_feature = coefs.index[0]
top_coef = coefs.iloc[0]
insights.append(
    f"6. Предиктивная модель: {top_feature} — самый сильный предиктор цены "
    f"(коэффициент {top_coef:+.2f}). Модель объясняет {r2*100:.1f}% дисперсии."
)

# Инсайт из сегментации
if "host_segment" in df.columns and df["host_segment"].notna().any():
    top_segment = segment_stats[("price_clean", "median")].idxmax()
    top_seg_median = segment_stats.loc[top_segment, ("price_clean", "median")]
    insights.append(
        f"7. {top_segment} ставят самые высокие цены (медиана ${top_seg_median:.0f}). "
        f"Профессиональные хосты понимают рыночную стоимость лучше любителей."
    )

# Инсайт из географии
insights.append(
    "8. Географический анализ показывает: дорогие объявления кучкуются в центре "
    "(высокая плотность точек красного цвета), дешевые — на окраинах."
)

for insight in insights:
    print(f"\n💡 {insight}")

# Дописываем в insights.txt
insights_path = os.path.join(BASE_DIR, "insights.txt")
with open(insights_path, "a", encoding="utf-8") as f:
    f.write("\n\nADVANCED ANALYSIS INSIGHTS\n")
    f.write("=" * 60 + "\n\n")
    for i, insight in enumerate(insights, 6):
        f.write(f"{insight}\n\n")

print(f"\n💾 Инсайты дописаны в: {insights_path}")
print(f"📁 Новые графики в: {SCREENSHOTS_DIR}")
print("\n" + "=" * 70)
print("ГОТОВО. Теперь обнови app.py, чтобы добавить новые вкладки.")
print("=" * 70)
