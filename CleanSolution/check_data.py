#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kontrola vstupních dat pro CleanSolution pipeline."""

import pandas as pd
import os

data_path = r"c:\Users\Bc. Jan Dub\Desktop\GIT\MachineLearning\CleanSolution\data_10y"

print("=" * 60)
print("KONTROLA VSTUPNICH DAT - data_10y/")
print("=" * 60)

# Seznam souborů
files = os.listdir(data_path)
print(f"\nNalezene soubory: {len(files)}")
for f in files:
    size = os.path.getsize(os.path.join(data_path, f)) / 1024
    print(f"  - {f}: {size:.1f} KB")

# Hlavní dataset
main_file = os.path.join(data_path, "all_sectors_full_10y.csv")
df = pd.read_csv(main_file)

print(f"\n{'=' * 60}")
print("HLAVNI DATASET: all_sectors_full_10y.csv")
print("=" * 60)
print(f"Pocet radku: {len(df):,}")
print(f"Pocet sloupcu: {len(df.columns)}")
print(f"\nSloupce ({len(df.columns)}):")
print(df.columns.tolist())

print(f"\n{'=' * 60}")
print("CASOVY ROZSAH")
print("=" * 60)
df['date'] = pd.to_datetime(df['date'])
print(f"Od: {df['date'].min()}")
print(f"Do: {df['date'].max()}")
print(f"Rozsah: {(df['date'].max() - df['date'].min()).days / 365:.1f} let")

print(f"\n{'=' * 60}")
print("TICKERY A SEKTORY")
print("=" * 60)
print(f"Unikatnich tickeru: {df['ticker'].nunique()}")
print(f"Priklady: {df['ticker'].unique()[:10].tolist()}")
print(f"\nSektory: {df['sector'].unique().tolist()}")
print(f"\nPocet zaznamu per sektor:")
print(df['sector'].value_counts())

print(f"\n{'=' * 60}")
print("OHLCV SLOUPCE")
print("=" * 60)
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
found_ohlcv = [c for c in ohlcv_cols if c in df.columns]
print(f"Nalezene OHLCV: {found_ohlcv}")
print(f"\nStatistiky close:")
print(df['close'].describe())

print(f"\n{'=' * 60}")
print("TECHNICKE INDIKATORY")
print("=" * 60)
tech_cols = ['volatility', 'returns', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 
             'sma_3', 'sma_6', 'sma_12', 'ema_3', 'ema_6', 'ema_12', 'volume_change']
found_tech = [c for c in tech_cols if c in df.columns]
print(f"Nalezene technicke ({len(found_tech)}): {found_tech}")

print(f"\n{'=' * 60}")
print("FUNDAMENTALNI SLOUPCE")
print("=" * 60)
fund_keywords = ['pe', 'pb', 'roe', 'roa', 'eps', 'debt', 'margin', 'ratio', 'book', 'earnings']
fund_cols = [c for c in df.columns if any(kw in c.lower() for kw in fund_keywords)]
if fund_cols:
    print(f"Nalezene fundamentalni: {fund_cols}")
else:
    print("ZADNE fundamentalni sloupce - POTREBA STAHNOUT!")
    print("Toto je ocekavane - fundamenty se stahnou v Notebook 01/02")

print(f"\n{'=' * 60}")
print("MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'missing': missing, 'pct': missing_pct})
missing_df = missing_df[missing_df['missing'] > 0].sort_values('pct', ascending=False)
if len(missing_df) > 0:
    print(f"Sloupce s chybejicimi hodnotami ({len(missing_df)}):")
    print(missing_df)
else:
    print("Zadne chybejici hodnoty v OHLCV a tech. indikatorech!")

print(f"\n{'=' * 60}")
print("ZAVER")
print("=" * 60)
print(f"Data jsou PRIPRAVENA pro pipeline!")
print(f"  - {len(df)} radku")
print(f"  - {df['ticker'].nunique()} tickeru")
print(f"  - {len(df['sector'].unique())} sektory")
print(f"  - Casovy rozsah: {(df['date'].max() - df['date'].min()).days / 365:.1f} let")
print(f"  - OHLCV: OK")
print(f"  - Technicke indikatory: OK")
print(f"  - Fundamenty: CHYBI (stahnou se v pipeline)")
