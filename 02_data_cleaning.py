#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02 — Data Cleaning
==================
Этот скрипт берет сырые объединенные данные и превращает их в чистую таблицу,
готовую для анализа. Что делаем:
1. Чистим цену (убираем $, запятые → число)
2. Убираем выбросы (абсурдно высокие/низкие цены)
3. Обрабатываем пропуски в рейтинге
4. Создаем новые признаки (категории цен, флаги)
5. Сохраняем результат
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# БЛОК 1: ПУТИ
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

input_path = os.path.join(PROCESSED_DIR, "airbnb_raw_combined.csv")
output_path = os.path.join(PROCESSED_DIR, "airbnb_cleaned.csv")

print("=" * 70)
print("ОЧИСТКА ДАННЫХ")
print("=" * 70)

# =============================================================================
# БЛОК 2: ЗАГРУЗКА
# =============================================================================

print(f"\n📂 Загружаем: {input_path}")
df = pd.read_csv(input_path, low_memory=False)
print(f"   Исходная таблица: {len(df):,} строк × {len(df.columns)} колонок")

# =============================================================================
# БЛОК 3: ЧИСТКА ЦЕНЫ
# =============================================================================

print("\n" + "-" * 70)
print("ШАГ 1: Чистка колонки price")
print("-" * 70)

# Смотрим, как выглядит цена до очистки
print(f"\nПример цен ДО: {df['price'].head(3).tolist()}")

# price выглядит как "$176.00" — это строка.
# Убираем знак $ и запятые, превращаем в float.
# str.replace('$', '', regex=False) — regex=False значит "ищи точное совпадение, не регулярку"
df["price_clean"] = (
    df["price"]
    .astype(str)                           # на всякий случай приводим к строке
    .str.replace("$", "", regex=False)    # убираем $
    .str.replace(",", "", regex=False)    # убираем запятые (если есть)
    .astype(float)                        # превращаем в число с плавающей точкой
)

print(f"Пример цен ПОСЛЕ: {df['price_clean'].head(3).tolist()}")

# =============================================================================
# БЛОК 4: УБИРАЕМ ВЫБРОСЫ ПО ЦЕНЕ
# =============================================================================

print("\n" + "-" * 70)
print("ШАГ 2: Удаление выбросов по цене")
print("-" * 70)

# Смотрим базовую статистику ДО удаления выбросов
print("\nСтатистика цены ДО:")
print(df["price_clean"].describe())

# Убираем объявления с ценой ниже $10 (ошибки/спам) и выше $2000 (люкс-виллы, не рынок)
# Эти пороги можно менять, но для анализа массового рынка они разумны.
MIN_PRICE = 10
MAX_PRICE = 2000

rows_before = len(df)
df = df[(df["price_clean"] >= MIN_PRICE) & (df["price_clean"] <= MAX_PRICE)]
rows_after = len(df)

print(f"\nУдалено строк: {rows_before - rows_after:,} ({(rows_before - rows_after) / rows_before * 100:.1f}%)")
print(f"Осталось: {rows_after:,} строк")

# =============================================================================
# БЛОК 5: ОБРАБОТКА ПРОПУСКОВ В РЕЙТИНГЕ
# =============================================================================

print("\n" + "-" * 70)
print("ШАГ 3: Обработка пропусков в review_scores_rating")
print("-" * 70)

# Смотрим, сколько пропусков
missing_before = df["review_scores_rating"].isna().sum()
print(f"\nПропусков в рейтинге: {missing_before:,} ({missing_before / len(df) * 100:.1f}%)")

# Стратегия: если у объявления 0 отзывов — рейтинг NaN логичен.
# Если отзывы есть, но рейтинг NaN — заполняем медианой по городу и типу жилья.
# Сначала создаем флаг: есть ли отзывы?
df["has_reviews"] = df["number_of_reviews"] > 0

# Для объявлений С отзывами, но БЕЗ рейтинга — заполняем медианой
df["review_scores_rating"] = df.groupby(["city", "room_type"])["review_scores_rating"].transform(
    lambda x: x.fillna(x.median())
)

# Если после группировки остались NaN (например, вся группа пустая) — заполняем общей медианой
df["review_scores_rating"] = df["review_scores_rating"].fillna(df["review_scores_rating"].median())

missing_after = df["review_scores_rating"].isna().sum()
print(f"Пропусков после заполнения: {missing_after}")

# =============================================================================
# БЛОК 6: СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ
# =============================================================================

print("\n" + "-" * 70)
print("ШАГ 4: Создание новых признаков")
print("-" * 70)

# --- 6.1 Категория цены ---
# Разбиваем цены на 3 группы. Пороги можно менять, но эти разумны для массового рынка.
def categorize_price(price):
    if price < 75:
        return "Budget"
    elif price <= 150:
        return "Mid"
    else:
        return "Luxury"

df["price_category"] = df["price_clean"].apply(categorize_price)
print("\n✅ Создана колонка: price_category (Budget / Mid / Luxury)")
print(df["price_category"].value_counts())

# --- 6.2 Флаг высокого рейтинга ---
# 4.8 и выше — это топ-15-20% объявлений. Хороший порог для "премиум качества".
df["is_highly_rated"] = df["review_scores_rating"] >= 4.8
print("\n✅ Создана колонка: is_highly_rated (True если рейтинг >= 4.8)")
print(df["is_highly_rated"].value_counts())

# --- 6.3 Цена на отзыв ---
# Индекс: сколько "платит" клиент за один отзыв других гостей.
# Чем меньше — тем популярнее (много отзывов при низкой цене).
# Осторожно с делением на 0: у объявлений без отзывов ставим NaN.
df["price_per_review"] = np.where(
    df["number_of_reviews"] > 0,
    df["price_clean"] / df["number_of_reviews"],
    np.nan
)
print("\n✅ Создана колонка: price_per_review (цена / количество отзывов)")

# --- 6.4 Средняя цена по городу (для контекста) ---
# Добавляем статистику города как признак — поможет в анализе
city_stats = df.groupby("city")["price_clean"].agg(["median", "mean"]).reset_index()
city_stats.columns = ["city", "city_price_median", "city_price_mean"]
df = df.merge(city_stats, on="city", how="left")
print("\n✅ Созданы колонки: city_price_median, city_price_mean")

# =============================================================================
# БЛОК 7: ФИНАЛЬНАЯ СТАТИСТИКА
# =============================================================================

print("\n" + "=" * 70)
print("ФИНАЛЬНАЯ СТАТИСТИКА ПО ГОРОДАМ")
print("=" * 70)

summary = df.groupby("city").agg({
    "price_clean": ["count", "median", "mean"],
    "review_scores_rating": "mean",
    "number_of_reviews": "median",
}).round(2)

print(summary)

# =============================================================================
# БЛОК 8: СОХРАНЕНИЕ
# =============================================================================

# Оставляем только нужные колонки для следующих этапов
# Это уменьшает размер файла и упрощает работу
COLUMNS_TO_KEEP = [
    "id", "city", "neighbourhood_cleansed", "room_type",
    "price_clean", "price_category",
    "number_of_reviews", "review_scores_rating", "has_reviews", "is_highly_rated",
    "price_per_review",
    "latitude", "longitude", "minimum_nights",
    "city_price_median", "city_price_mean"
]

df_final = df[COLUMNS_TO_KEEP].copy()

df_final.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print(f"💾 Сохранено: {output_path}")
print(f"   Строк: {len(df_final):,}")
print(f"   Колонок: {len(df_final.columns)}")
print(f"   Размер: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
print("=" * 70)
print("\nГОТОВО. Переходи к 03_eda.py")
