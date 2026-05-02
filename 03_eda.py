#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03 — Exploratory Data Analysis (EDA)
=====================================
Этот скрипт строит графики и находит ключевые инсайты.
Результат: 6–8 графиков в папке screenshots/ + текстовый отчет с выводами.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройка стиля графиков
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# =============================================================================
# БЛОК 1: ПУТИ
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

input_path = os.path.join(PROCESSED_DIR, "airbnb_cleaned.csv")

print("=" * 70)
print("ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ (EDA)")
print("=" * 70)

# =============================================================================
# БЛОК 2: ЗАГРУЗКА
# =============================================================================

print(f"\n📂 Загружаем: {input_path}")
df = pd.read_csv(input_path)
print(f"   Таблица: {len(df):,} строк × {len(df.columns)} колонок")

# =============================================================================
# БЛОК 3: ГРАФИК 1 — Распределение цен по городам (Boxplot)
# =============================================================================

print("\n📊 Строим график 1: Распределение цен по городам...")

fig, ax = plt.subplots(figsize=(10, 6))
# Ограничиваем ось Y до $500, чтобы выбросы не сжимали график
sns.boxplot(data=df[df["price_clean"] <= 500], x="city", y="price_clean", ax=ax)
ax.set_title("Price Distribution by City (up to $500)", fontsize=14, fontweight="bold")
ax.set_xlabel("City", fontsize=12)
ax.set_ylabel("Price per Night ($)", fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "01_price_distribution_boxplot.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 4: ГРАФИК 2 — Средняя и медианная цена по городам
# =============================================================================

print("📊 Строим график 2: Средняя и медианная цена...")

city_stats = df.groupby("city")["price_clean"].agg(["mean", "median"]).reset_index()
city_stats = city_stats.sort_values("median", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(city_stats))
width = 0.35

bars1 = ax.bar(x - width/2, city_stats["mean"], width, label="Mean", alpha=0.8)
bars2 = ax.bar(x + width/2, city_stats["median"], width, label="Median", alpha=0.8)

ax.set_title("Average vs Median Price by City", fontsize=14, fontweight="bold")
ax.set_xlabel("City", fontsize=12)
ax.set_ylabel("Price ($)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(city_stats["city"], rotation=15)
ax.legend()

# Добавляем значения на столбцы
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f"${height:.0f}", xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f"${height:.0f}", xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "02_mean_median_price.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 5: ГРАФИК 3 — Тип жилья vs цена
# =============================================================================

print("📊 Строим график 3: Тип жилья vs цена...")

fig, ax = plt.subplots(figsize=(10, 6))
# Берем только топ-3 типа жилья (их 99% данных)
top_room_types = df["room_type"].value_counts().head(3).index.tolist()
df_room = df[df["room_type"].isin(top_room_types) & (df["price_clean"] <= 400)]

sns.boxplot(data=df_room, x="room_type", y="price_clean", ax=ax)
ax.set_title("Price by Room Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Room Type", fontsize=12)
ax.set_ylabel("Price per Night ($)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "03_price_by_room_type.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 6: ГРАФИК 4 — Рейтинг vs цена (Scatter)
# =============================================================================

print("📊 Строим график 4: Рейтинг vs цена...")

# Берем выборку, чтобы график не тормозил (10 000 точек достаточно)
df_sample = df.sample(n=min(10000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df_sample, x="review_scores_rating", y="price_clean",
                hue="city", alpha=0.5, ax=ax)
ax.set_title("Price vs Review Rating", fontsize=14, fontweight="bold")
ax.set_xlabel("Review Score (0–5)", fontsize=12)
ax.set_ylabel("Price per Night ($)", fontsize=12)
ax.set_ylim(0, 500)  # обрезаем выбросы для наглядности
plt.legend(title="City", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "04_price_vs_rating_scatter.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 7: ГРАФИК 5 — Топ-10 районов по средней цене (для каждого города)
# =============================================================================

print("📊 Строим график 5: Топ-10 дорогих районов...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

cities = df["city"].unique()
for i, city in enumerate(cities):
    ax = axes[i]
    city_df = df[df["city"] == city]
    top_neigh = city_df.groupby("neighbourhood_cleansed")["price_clean"].mean().sort_values(ascending=False).head(10)
    top_neigh.plot(kind="barh", ax=ax, color=sns.color_palette("husl", 10))
    ax.set_title(f"{city} — Top 10 Neighborhoods by Avg Price", fontsize=12, fontweight="bold")
    ax.set_xlabel("Average Price ($)", fontsize=10)
    ax.invert_yaxis()  # самый дорогой сверху

# Убираем лишний subplot
axes[5].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "05_top_neighborhoods_by_city.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 8: ГРАФИК 6 — Доля высокорейтинговых объявлений по городам
# =============================================================================

print("📊 Строим график 6: Доля высокорейтинговых объявлений...")

rating_by_city = df.groupby("city")["is_highly_rated"].mean().sort_values(ascending=False).reset_index()
rating_by_city["is_highly_rated"] *= 100  # в процентах

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(rating_by_city["city"], rating_by_city["is_highly_rated"], color=sns.color_palette("husl", 5))
ax.set_title("Share of Highly Rated Listings (≥4.8) by City", fontsize=14, fontweight="bold")
ax.set_xlabel("City", fontsize=12)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "06_highly_rated_share.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 9: ГРАФИК 7 — Распределение по категориям цен
# =============================================================================

print("📊 Строим график 7: Распределение по категориям цен...")

price_cat = df.groupby(["city", "price_category"]).size().unstack(fill_value=0)
price_cat_pct = price_cat.div(price_cat.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 6))
price_cat_pct.plot(kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"])
ax.set_title("Price Category Distribution by City (%)", fontsize=14, fontweight="bold")
ax.set_xlabel("City", fontsize=12)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.legend(title="Price Category", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "07_price_category_distribution.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 10: КЛЮЧЕВЫЕ ИНСАЙТЫ
# =============================================================================

print("\n" + "=" * 70)
print("КЛЮЧЕВЫЕ ИНСАЙТЫ")
print("=" * 70)

insights = []

# Инсайт 1: Самый дорогой город
most_expensive = city_stats.iloc[0]["city"]
most_expensive_median = city_stats.iloc[0]["median"]
insights.append(
    f"1. {most_expensive} — самый дорогой рынок (медиана ${most_expensive_median:.0f}/ночь). "
    f"При этом среднее (${city_stats.iloc[0]['mean']:.0f}) сильно выше медианы — "
    f"значит, много супер-люкс объектов, которые тянут среднее вверх."
)

# Инсайт 2: NYC — объем и качество
cheapest_median = city_stats.sort_values("median").iloc[0]
insights.append(
    f"2. {cheapest_median['city']} — самый доступный рынок (медиана ${cheapest_median['median']:.0f}), "
    f"но и самый низкий рейтинг ({df[df['city']==cheapest_median['city']]['review_scores_rating'].mean():.2f}). "
    f"Возможно, перенасыщение рынка снижает качество обслуживания."
)

# Инсайт 3: Разница между mean и median
max_gap_city = city_stats.loc[(city_stats["mean"] - city_stats["median"]).idxmax()]
insights.append(
    f"3. {max_gap_city['city']} имеет самый большой разрыв между средней и медианной ценой "
    f"(${max_gap_city['mean']:.0f} vs ${max_gap_city['median']:.0f}). "
    f"Это указывает на высокую дисперсию: есть массовый сегмент и элитный сегмент, но мало среднего."
)

# Инсайт 4: Рейтинг vs цена
high_rated_expensive = df[df["is_highly_rated"]].groupby("city")["price_clean"].median().sort_values(ascending=False)
insights.append(
    f"4. Среди высокорейтинговых объявлений (≥4.8) самые дорогие в {high_rated_expensive.index[0]} "
    f"(${high_rated_expensive.iloc[0]:.0f}). Это говорит о том, что высокое качество там стоит денег."
)

# Инсайт 5: Доля люкса
luxury_share = df.groupby("city")["price_category"].apply(lambda x: (x == "Luxury").mean() * 100).sort_values(ascending=False)
insights.append(
    f"5. {luxury_share.index[0]} — лидер по доле люкс-объявлений ({luxury_share.iloc[0]:.1f}%). "
    f"Это рынок с высоким потенциалом для инвесторов в премиум-недвижимость."
)

for insight in insights:
    print(f"\n💡 {insight}")

# =============================================================================
# БЛОК 11: СОХРАНЕНИЕ ИНСАЙТОВ В ФАЙЛ
# =============================================================================

insights_path = os.path.join(BASE_DIR, "insights.txt")
with open(insights_path, "w", encoding="utf-8") as f:
    f.write("AIRBNB PRICING ANALYSIS — KEY INSIGHTS\n")
    f.write("=" * 60 + "\n\n")
    for i, insight in enumerate(insights, 1):
        f.write(f"{insight}\n\n")

print(f"\n💾 Инсайты сохранены: {insights_path}")
print(f"📁 Графики сохранены в: {SCREENSHOTS_DIR}")
print("\n" + "=" * 70)
print("ГОТОВО. Переходи к app.py (Streamlit Dashboard)")
print("=" * 70)
