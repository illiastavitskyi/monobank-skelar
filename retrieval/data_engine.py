"""Loads dataset and computes per-category stats."""

import json
import os
import pandas as pd
import numpy as np

import config


class DataEngine:
    def __init__(self):
        self.df: pd.DataFrame = pd.DataFrame()
        self.category_dict: dict[str, str] = {}
        self.category_stats: dict[str, dict] = {}

    def load(self) -> None:
        self.category_dict = self._load_category_dict()
        self.df = self._load_advertisements()
        self.category_stats = self._compute_or_load_stats()

    def _load_category_dict(self) -> dict[str, str]:
        with open(config.CATEGORY_DICT, encoding="utf-8") as f:
            return json.load(f)

    def _load_advertisements(self) -> pd.DataFrame:
        df = pd.read_csv(config.ADVERTISEMENTS_CSV)
        df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")
        df["sold_price"] = pd.to_numeric(df["sold_price"], errors="coerce")
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["modified_at"] = pd.to_datetime(df["modified_at"], errors="coerce")
        df["category_id"] = df["category_id"].astype(str)
        df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.strip()
        return df

    def _compute_or_load_stats(self) -> dict[str, dict]:
        if os.path.exists(config.CATEGORY_STATS_CACHE):
            with open(config.CATEGORY_STATS_CACHE, encoding="utf-8") as f:
                return json.load(f)
        stats = self._compute_stats()
        with open(config.CATEGORY_STATS_CACHE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return stats

    def _compute_stats(self) -> dict[str, dict]:
        sold = self.df[self.df["status"] == "SOLD"].copy()
        sold["days_to_sell"] = (sold["modified_at"] - sold["created_at"]).dt.days.clip(lower=0)
        sold["bargain_discount"] = (
            (sold["original_price"] - sold["sold_price"]) / sold["original_price"]
        ).clip(lower=0, upper=1)

        stats: dict[str, dict] = {}
        for cat_id, group in sold.groupby("category_id"):
            prices = group["sold_price"].dropna()
            days = group["days_to_sell"].dropna()
            bargain = group[group["sold_via_bargain"] == True]["bargain_discount"].dropna()  # noqa: E712
            total = len(self.df[self.df["category_id"] == cat_id])
            sold_count = len(group)

            stats[str(cat_id)] = {
                "name": self.category_dict.get(str(cat_id), str(cat_id)),
                "total_listings": int(total),
                "sold_count": int(sold_count),
                "sell_rate": round(sold_count / total, 3) if total else 0,
                "price_p20": round(float(np.percentile(prices, 20)), 2) if len(prices) else 0,
                "price_p35": round(float(np.percentile(prices, 35)), 2) if len(prices) else 0,
                "price_p50": round(float(np.percentile(prices, 50)), 2) if len(prices) else 0,
                "price_p60": round(float(np.percentile(prices, 60)), 2) if len(prices) else 0,
                "price_p70": round(float(np.percentile(prices, 70)), 2) if len(prices) else 0,
                "price_p85": round(float(np.percentile(prices, 85)), 2) if len(prices) else 0,
                "median_days_to_sell": round(float(days.median()), 1) if len(days) else None,
                "avg_bargain_discount_pct": round(float(bargain.mean()) * 100, 1) if len(bargain) else 0,
            }
        return stats

    def get_category_stats(self, category_id: str) -> dict | None:
        return self.category_stats.get(str(category_id))
