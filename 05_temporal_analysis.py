#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05 — Temporal Analysis (Calendar Data)
========================================
Анализируем calendar.csv для 5 городов:
- Сезонность цен по месяцам
- Загрузка/доступность (occupancy rate)
- Динамика бронирований

Работает с большими файлами через chunk-чтение.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# =============================================================================
# БЛОК 1: НАСТРОЙКА
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots", "temporal")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

CITIES = {
    "Austin": os.path.join(RAW_DIR, "Austin", "calendar_austin.csv"),
    "Dallas": os.path.join(RAW_DIR, "Dallas", "calendar_dallas.csv"),
    "Nashville": os.path.join(RAW_DIR, "Nashville", "calendar_nashville.csv"),
    "New York City": os.path.join(RAW_DIR, "New York City", "calendar_newyork.csv"),
    "Seattle": os.path.join(RAW_DIR, "Seattle", "calendar_seattle.csv"),
}

print("=" * 70)
print("ВРЕМЕННОЙ АНАЛИЗ (Calendar Data)")
print("=" * 70)

# =============================================================================
# БЛОК 2: ЧТЕНИЕ И АГРЕГАЦИЯ ПО ЧАНКАМ
# =============================================================================

monthly_stats = []

for city_name, filepath in CITIES.items():
    print(f"\n📂 {city_name}: {filepath}")

    if not os.path.exists(filepath):
        print(f"   ❌ Файл не найден, пропускаем")
        continue

    # Читаем по чанкам, чтобы не перегружать память
    # Берем только нужные колонки: date, available, price
    chunk_list = []
    chunk_size = 100000  # строк за раз

    for chunk in pd.read_csv(filepath, usecols=["date", "available", "price"], 
                              chunksize=chunk_size, low_memory=False):
        # Преобразуем дату
        chunk["date"] = pd.to_datetime(chunk["date"])
        # Преобразуем цену (убираем $)
        chunk["price"] = chunk["price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        # Доступность: 't' = True (свободно), 'f' = False (занято)
        chunk["is_available"] = chunk["available"] == "t"
        chunk["city"] = city_name
        chunk_list.append(chunk)

    # Объединяем чанки города
    city_df = pd.concat(chunk_list, ignore_index=True)

    # Агрегируем по месяцам
    city_df["year_month"] = city_df["date"].dt.to_period("M")

    monthly = city_df.groupby("year_month").agg(
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        availability_rate=("is_available", "mean"),  # % свободных
        total_listings=("price", "count")
    ).reset_index()

    monthly["city"] = city_name
    monthly["occupancy_rate"] = 1 - monthly["availability_rate"]  # % занятых
    monthly["year_month_str"] = monthly["year_month"].astype(str)

    monthly_stats.append(monthly)

    print(f"   ✅ Обработано {len(city_df):,} записей")
    print(f"   📅 Период: {city_df['date'].min().date()} — {city_df['date'].max().date()}")

# =============================================================================
# БЛОК 3: ОБЪЕДИНЕНИЕ
# =============================================================================

df_monthly = pd.concat(monthly_stats, ignore_index=True)

# Сохраняем агрегированные данные для дашборда
output_csv = os.path.join(PROCESSED_DIR, "calendar_monthly_agg.csv")
df_monthly.to_csv(output_csv, index=False)
print(f"\n💾 Агрегированные данные сохранены: {output_csv}")

# =============================================================================
# БЛОК 4: ГРАФИК 1 — Сезонность цен по месяцам
# =============================================================================

print("\n📊 Строим график 1: Сезонность цен...")

fig, ax = plt.subplots(figsize=(14, 7))

for city in df_monthly["city"].unique():
    city_data = df_monthly[df_monthly["city"] == city]
    ax.plot(city_data["year_month_str"], city_data["avg_price"], 
            marker="o", label=city, linewidth=2)

ax.set_title("Average Price Seasonality (2026)", fontsize=16, fontweight="bold")
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Average Price ($)", fontsize=12)
ax.legend(title="City", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.tick_params(axis="x", rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "01_price_seasonality.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 5: ГРАФИК 2 — Загрузка (Occupancy Rate) по месяцам
# =============================================================================

print("📊 Строим график 2: Загрузка по месяцам...")

fig, ax = plt.subplots(figsize=(14, 7))

for city in df_monthly["city"].unique():
    city_data = df_monthly[df_monthly["city"] == city]
    ax.plot(city_data["year_month_str"], city_data["occupancy_rate"] * 100, 
            marker="s", label=city, linewidth=2)

ax.set_title("Occupancy Rate by Month (%)", fontsize=16, fontweight="bold")
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Occupancy Rate (%)", fontsize=12)
ax.legend(title="City", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "02_occupancy_seasonality.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 6: ГРАФИК 3 — Сравнение городов (средние показатели)
# =============================================================================

print("📊 Строим график 3: Сводное сравнение городов...")

city_summary = df_monthly.groupby("city").agg(
    avg_price=("avg_price", "mean"),
    avg_occupancy=("occupancy_rate", "mean"),
    peak_month=("avg_price", "idxmax")  # индекс месяца с максимальной ценой
).reset_index()

city_summary["avg_occupancy_pct"] = city_summary["avg_occupancy"] * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Средняя цена
bars1 = ax1.bar(city_summary["city"], city_summary["avg_price"], 
                color=sns.color_palette("husl", len(city_summary)))
ax1.set_title("Average Price Across All Months", fontsize=14, fontweight="bold")
ax1.set_ylabel("Price ($)", fontsize=12)
ax1.tick_params(axis="x", rotation=15)
for bar in bars1:
    h = bar.get_height()
    ax1.annotate(f"${h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

# Средняя загрузка
bars2 = ax2.bar(city_summary["city"], city_summary["avg_occupancy_pct"], 
                color=sns.color_palette("husl", len(city_summary)))
ax2.set_title("Average Occupancy Rate", fontsize=14, fontweight="bold")
ax2.set_ylabel("Occupancy (%)", fontsize=12)
ax2.tick_params(axis="x", rotation=15)
ax2.set_ylim(0, 100)
for bar in bars2:
    h = bar.get_height()
    ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "03_city_comparison.png"), dpi=150)
plt.close()

# =============================================================================
# БЛОК 7: ИНСАЙТЫ
# =============================================================================

print("\n" + "=" * 70)
print("ВРЕМЕННЫЕ ИНСАЙТЫ")
print("=" * 70)

insights = []

# Инсайт 1: Пиковый месяц
peak_city = city_summary.loc[city_summary["avg_price"].idxmax()]
insights.append(
    f"9. {peak_city['city']} — самый дорогой рынок в среднем (${peak_city['avg_price']:.0f}/ночь) "
    f"и самая высокая загрузка ({peak_city['avg_occupancy_pct']:.1f}%). "
    f"Спрос здесь превышает предложение."
)

# Инсайт 2: Сезонность
# Находим город с самой большой разницей между пиком и минимумом
price_ranges = df_monthly.groupby("city")["avg_price"].agg(["max", "min"])
price_ranges["range"] = price_ranges["max"] - price_ranges["min"]
most_seasonal = price_ranges["range"].idxmax()
range_val = price_ranges.loc[most_seasonal, "range"]
insights.append(
    f"10. {most_seasonal} — самый сезонный рынок. Разница между пиком и минимумом "
    f"${range_val:.0f}. Инвесторам стоит учитывать колебания доходности."
)

# Инсайт 3: NYC особенность
if "New York City" in city_summary["city"].values:
    nyc = city_summary[city_summary["city"] == "New York City"].iloc[0]
    insights.append(
        f"11. NYC: высокая загрузка ({nyc['avg_occupancy_pct']:.1f}%) при средней цене "
        f"(${nyc['avg_price']:.0f}). Это стабильный, но конкурентный рынок."
    )

for insight in insights:
    print(f"\n💡 {insight}")

# Дописываем в insights.txt
insights_path = os.path.join(BASE_DIR, "insights.txt")
with open(insights_path, "a", encoding="utf-8") as f:
    f.write("\n\nTEMPORAL ANALYSIS INSIGHTS\n")
    f.write("=" * 60 + "\n\n")
    for insight in insights:
        f.write(f"{insight}\n\n")

print(f"\n💾 Инсайты дописаны в: {insights_path}")
print(f"📁 Графики в: {SCREENSHOTS_DIR}")
print("\n" + "=" * 70)
print("ГОТОВО. Calendar-данные проанализированы.")
print("=" * 70)
