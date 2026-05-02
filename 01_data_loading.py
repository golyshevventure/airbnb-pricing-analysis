#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01 — Data Loading
=================
Этот скрипт загружает CSV-файлы Airbnb для 5 американских городов,
проверяет их структуру, качество данных и объединяет в единую таблицу.

Города: Austin, Dallas, Nashville, New York City, Seattle
"""

import pandas as pd
import os

# =============================================================================
# БЛОК 1: НАСТРОЙКА ПУТЕЙ
# =============================================================================

# Определяем корневую папку проекта.
# __file__ — это путь к текущему скрипту.
# os.path.dirname(__file__) — папка, где лежит скрипт.
# Это позволяет запускать скрипт из любой папки — пути всегда будут правильными.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папка с сырыми данными
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Папка для обработанных данных
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Создаем папку processed, если её нет
os.makedirs(PROCESSED_DIR, exist_ok=True)

# =============================================================================
# БЛОК 2: КОНФИГУРАЦИЯ ГОРОДОВ
# =============================================================================

# Словарь: название города → путь к CSV-файлу
# Используем os.path.join вместо слэшей — это работает и на Windows, и на Linux.
CITIES = {
    "Austin": os.path.join(RAW_DIR, "Austin", "listings_austin.csv"),
    "Dallas": os.path.join(RAW_DIR, "Dallas", "listings_dallas.csv"),
    "Nashville": os.path.join(RAW_DIR, "Nashville", "listings_nashville.csv"),
    "New York City": os.path.join(RAW_DIR, "New York City", "listings_newyork.csv"),
    "Seattle": os.path.join(RAW_DIR, "Seattle", "listings_seattle.csv"),
}

# Колонки, которые нам нужны для проекта.
# Если какой-то из этих колонок нет в CSV — скрипт предупредит.
REQUIRED_COLS = [
    "price",                  # цена за ночь (строка с $)
    "neighbourhood_cleansed", # район
    "room_type",              # тип жилья
    "number_of_reviews",      # количество отзывов
    "review_scores_rating",   # средний рейтинг (0–5)
    "latitude",               # широта
    "longitude",              # долгота
    "minimum_nights",         # минимальное количество ночей
]

# =============================================================================
# БЛОК 3: ЗАГРУЗКА ДАННЫХ
# =============================================================================

# Словарь для хранения DataFrame каждого города
dfs = {}

print("=" * 70)
print("ЗАГРУЗКА ДАННЫХ AIRBNB")
print("=" * 70)
print()

for city_name, filepath in CITIES.items():
    print(f"\n📂 Город: {city_name}")
    print(f"   Путь: {filepath}")

    # Проверяем, существует ли файл
    if not os.path.exists(filepath):
        print(f"   ❌ Файл не найден! Пропускаем.")
        continue

    # Читаем CSV.
    # low_memory=False — отключает предупреждения о смешанных типах данных.
    df = pd.read_csv(filepath, low_memory=False)

    # Добавляем колонку с названием города (чтобы потом различать)
    df["city"] = city_name

    # Сохраняем в словарь
    dfs[city_name] = df

    print(f"   ✅ Загружено: {len(df):,} строк, {len(df.columns)} колонок")
    print(f"   📊 Размер файла: {os.path.getsize(filepath) / (1024*1024):.1f} MB")

print("\n" + "=" * 70)

# =============================================================================
# БЛОК 4: ПРОВЕРКА КЛЮЧЕВЫХ КОЛОНОК
# =============================================================================

print("\nПРОВЕРКА КЛЮЧЕВЫХ КОЛОНОК")
print("-" * 70)

for city_name, df in dfs.items():
    print(f"\n🏙️  {city_name}:")
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        print(f"   ⚠️  Отсутствуют колонки: {missing}")
    else:
        print(f"   ✅ Все ключевые колонки на месте")

    # Показываем первые 3 колонки для контекста
    print(f"   📋 Все колонки: {list(df.columns[:8])}...")

# =============================================================================
# БЛОК 5: ПЕРВЫЙ ВЗГЛЯД НА ДАННЫЕ
# =============================================================================

print("\n" + "=" * 70)
print("ПЕРВЫЙ ВЗГЛЯД НА ДАННЫЕ")
print("=" * 70)

for city_name, df in dfs.items():
    print(f"\n🏙️  {city_name} — первые 3 строки:")
    # Показываем только ключевые колонки (если они есть)
    cols_to_show = [col for col in REQUIRED_COLS if col in df.columns]
    print(df[cols_to_show].head(3).to_string())

# =============================================================================
# БЛОК 6: ОБЪЕДИНЕНИЕ В ОДНУ ТАБЛИЦУ
# =============================================================================

print("\n" + "=" * 70)
print("ОБЪЕДИНЕНИЕ ДАННЫХ")
print("=" * 70)

# pd.concat склеивает список DataFrame один под другим
# ignore_index=True — пересчитывает индексы заново (0, 1, 2, ...)
df_combined = pd.concat(dfs.values(), ignore_index=True)

print(f"\n📦 Общая таблица: {len(df_combined):,} строк × {len(df_combined.columns)} колонок")
print(f"   Города: {df_combined['city'].value_counts().to_dict()}")

# =============================================================================
# БЛОК 7: СОХРАНЕНИЕ
# =============================================================================

output_path = os.path.join(PROCESSED_DIR, "airbnb_raw_combined.csv")
df_combined.to_csv(output_path, index=False)

print(f"\n💾 Сохранено: {output_path}")
print(f"   Размер: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

print("\n" + "=" * 70)
print("ГОТОВО. Переходи к 02_data_cleaning.py")
print("=" * 70)
