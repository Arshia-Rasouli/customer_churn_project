"""
app.py - سامانه تحلیل هوشمند و مدیریت ریزش مشتری (Phase 10: Persian presentation layer, new visual identity)

Only the presentation layer changes in this version. The engine files are used exactly as they are and
receive the RAW values of every column (nothing is converted or renamed before prediction):
    churn_model.joblib      -> trained model bundle
    churn_inference.py      -> predict_single_customer(customer_dict)
    test_stream_data.csv    -> customers used as the live stream

Display-only conversions (never sent to the engine):
    * PreferredPaymentMode  -> Persian labels (PAYMENT_FA)
    * CashbackAmount / avg_cashbk_per_order -> Toman = raw * 2000

Run (all files in the same folder):
    pip install "streamlit>=1.40" plotly pandas numpy scikit-learn joblib
    streamlit run app.py
Keep the accompanying .streamlit/config.toml next to this file.
"""

from __future__ import annotations

import html
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="سامانه تحلیل هوشمند و مدیریت ریزش مشتری",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Paths and design tokens
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
STREAM_PATH = DATA_DIR / "test_stream_data.csv"
INFERENCE_PATH = BASE_DIR / "churn_inference.py"

BG, BLUE, NAVY = "#C4D5F9", "#3157B7", "#1E3A73"
INK, MUTED, TEAL, BORDER = "#172554", "#64748B", "#2AAFA3", "#D6E1F5"
LOW, MID, HIGH = "#22A06B", "#E6A23C", "#D9534F"          # risk colours only
NEUTRAL_BAR = "#B9C9EE"
TONE_COLOR = {"red": HIGH, "amber": MID, "green": LOW}
RED_FROM, AMBER_FROM = 70.0, 35.0  # risk >= 70 high, 35..70 medium, < 35 stable

RADAR_SIZE = 2        # the radar always monitors exactly two customers
AUTO_EVERY = 6        # seconds between automatic radar swaps
TOMAN_PER_UNIT = 2    # display only: raw cashback value * 2000 (see cash_toman)

VIEW_BI = "تحلیل و بینش کسب‌وکار"
VIEW_LAB = " پایش و پیش‌بینی مشتری"
VIEWS = [VIEW_BI, VIEW_LAB]

DEFAULT_STORE = "کسب‌وکار شما"

CATEGORY_FA = {
    "Mobile Phone": "موبایل و دیجیتال", "Mobile": "موبایل و دیجیتال",
    "Laptop & Accessory": "لپ‌تاپ و تجهیزات", "Laptop and Accessory": "لپ‌تاپ و تجهیزات",
    "Fashion": "مد و پوشاک", "Grocery": "خواربار", "Others": "سایر",
}
# Display mapping for PreferredPaymentMode (raw stream spellings CC / COD included). Display only.
PAYMENT_FA = {
    "Debit Card": "کارت بانکی", "Credit Card": "کارت بانکی", "CC": "کارت بانکی",
    "E wallet": "کیف پول الکترونیکی", "Cash on Delivery": "پرداخت در محل", "COD": "پرداخت در محل",
    "UPI": "درگاه اینترنتی",
}
COLUMN_FA = {"PreferredLoginDevice": "دستگاه ورود", "PreferredPaymentMode": "روش پرداخت", "Gender": "جنسیت",
             "PreferedOrderCat": "دسته خرید", "MaritalStatus": "وضعیت تأهل"}
VALUE_FA = {
    "Mobile Phone": "موبایل", "Phone": "تلفن", "Computer": "رایانه", "Male": "مرد", "Female": "زن",
    "Single": "مجرد", "Married": "متأهل", "Divorced": "جداشده", "Laptop & Accessory": "لپ‌تاپ و تجهیزات",
    "Fashion": "مد و پوشاک", "Grocery": "خواربار", "Others": "سایر",
}

# ---- Retention-lab inputs (simple choices -> values understood by the engine) ----
LAB_TENURE = {"ماه اول": 1, "زیر ۶ ماه": 4, "۱ سال": 12, "بیش از ۲ سال": 30}           # months
LAB_DAYS = {"۳ روز پیش": 3, "۱۰ روز پیش": 10, "۲۰ روز پیش": 20}                        # days since last order
LAB_CATEGORIES = {"دیجیتال": "Mobile Phone", "مد و پوشاک": "Fashion", "سوپرمارکت": "Grocery",
                  "لپ‌تاپ": "Laptop & Accessory", "سایر": "Others"}
# ILLUSTRATIVE ESTIMATE ONLY (the data has no revenue column): assumed average annual purchase value of a
# customer per category, in Toman. The displayed retention value = annual value x churn probability.
CATEGORY_ANNUAL_VALUE = {"Mobile Phone": 60_000_000, "Laptop & Accessory": 90_000_000, "Fashion": 40_500_000,
                         "Grocery": 30_000_000, "Others": 30_500_000}
# Neutral defaults for the factors the visitor does not enter (numeric ones are median-imputed inside
# churn_inference; these categorical ones are the most frequent values of the data).
NEUTRAL_CATEGORICALS = {"PreferredLoginDevice": "Mobile Phone", "PreferredPaymentMode": "Debit Card",
                        "Gender": "Male", "MaritalStatus": "Married"}
MEDIAN_DAYS = 3

# Facts measured on the full 5,630-customer history (median-imputed). Cashback values are raw units (x2000 = Toman).
DATASET = {
    "n": 5630, "churned": 948, "retained": 4682, "churn_rate": 16.84, "detection_power": 99.8, "accuracy": 98,
    "satisfaction": {1: 11.5, 2: 12.6, 3: 17.2, 4: 17.1, 5: 23.8},
    "complain": {"بدون شکایت": 10.9, "دارای شکایت": 31.7}, "complain_share_of_churners": 53.6, "complain_customers": 1604,
    "tenure": {"۰ تا ۱ ماه": 51.8, "۲ تا ۳ ماه": 8.8, "۴ تا ۶ ماه": 7.5, "۷ تا ۱۲ ماه": 9.8, "بیش از ۱۲ ماه": 5.0},
    "tenure_mean_loyal": 11.4, "tenure_mean_churned": 3.9, "first_month_share_of_churners": 65.5,
    "category": {"Mobile Phone": 27.4, "Fashion": 15.5, "Laptop & Accessory": 10.2, "Others": 7.6, "Grocery": 4.9},
    "cashback_rate": [29.5, 20.3, 12.2, 10.2],      # churn % for raw bands <=130, 131-160, 161-200, >200
    "cashback_bounds_raw": [130, 160, 200],
    "cashback_mean_loyal_raw": 178.4, "cashback_mean_churned_raw": 159.6,
    "distance": {"تا ۱۰": 13.5, "۱۱ تا ۲۰": 17.6, "بیش از ۲۰": 20.4},
    "recency": {"۰ تا ۲ روز": 24.9, "بیش از ۱۰ روز": 6.2},
}

# ---------------------------------------------------------------------------------------------- #
# Retention lab (Tab 2) additions. Display / input layer only: the engine still receives RAW values.
# ---------------------------------------------------------------------------------------------- #
LAB_CATEGORY_LABELS = {"موبایل و دیجیتال": "Mobile Phone", "مد و پوشاک": "Fashion", "سوپرمارکت و خواربار": "Grocery",
                       "لپ‌تاپ و تجهیزات": "Laptop & Accessory", "سایر": "Others"}
LAB_TIER_LABELS = {"کلان‌شهرها (تهران، مشهد، اصفهان، شیراز، تبریز)": 1,   # CityTier 1
                   "مراکز استان و شهرهای بزرگ": 2,                        # CityTier 2
                   "سایر شهرستان‌ها و مناطق": 3}                          # CityTier 3
_T1, _T2, _T3 = list(LAB_TIER_LABELS)
LAB_DEFAULTS = {"lab_tenure_n": 2, "lab_complain": "خیر", "lab_cat_l": "موبایل و دیجیتال", "lab_days_n": 3,
                "lab_tier_l": _T1, "lab_sat": 3, "lab_orders_n": 2}
LAB_PRESETS = {  # quick scenarios: they only FILL the form; the visitor still presses the calculate button
    1: {"label": "خریدار شاکی",
        "values": {"lab_tenure_n": 1, "lab_complain": "بله", "lab_cat_l": "موبایل و دیجیتال", "lab_days_n": 3,
                   "lab_tier_l": _T1, "lab_sat": 3, "lab_orders_n": 2}},
    2: {"label": "مشتری پایدار",
        "values": {"lab_tenure_n": 36, "lab_complain": "خیر", "lab_cat_l": "سوپرمارکت و خواربار", "lab_days_n": 12,
                   "lab_tier_l": _T1, "lab_sat": 3, "lab_orders_n": 4}},
    3: {"label": "کاربر غیرفعال",
        "values": {"lab_tenure_n": 1, "lab_complain": "خیر", "lab_cat_l": "مد و پوشاک", "lab_days_n": 45,
                   "lab_tier_l": _T3, "lab_sat": 3, "lab_orders_n": 2}},
}

# The four lab inputs do not cover every column of the engine. The remaining raw fields are filled with the
# MEDIAN (numbers) / MOST FREQUENT value (categories) of the historical customers who share the same category,
# complaint status and tenure band (new: <= 1 month, mid: 2-12 months, old: > 12 months). Segments with fewer
# than 20 customers fall back to the category + complaint group, then to the overall medians.
PROFILE_FIELDS = ["CityTier", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered", "SatisfactionScore",
                  "NumberOfAddress", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "CashbackAmount",
                  "PreferredLoginDevice", "PreferredPaymentMode", "Gender", "MaritalStatus"]
SEGMENT_PROFILES = {
    ('Mobile Phone', 0, 'new'): (1.0, 14.0, 3.0, 4.0, 3.0, 3.0, 15.0, 1.0, 2.0, 144.0, 'Phone', 'Credit Card', 'Male', 'Single'),
    ('Mobile Phone', 0, 'mid'): (1.0, 13.0, 3.0, 4.0, 3.0, 3.0, 15.0, 1.0, 2.0, 137.0, 'Phone', 'Debit Card', 'Male', 'Married'),
    ('Mobile Phone', 0, 'old'): (1.0, 14.0, 3.0, 4.0, 3.0, 5.0, 15.0, 1.0, 2.0, 148.0, 'Phone', 'Debit Card', 'Male', 'Married'),
    ('Mobile Phone', 1, 'new'): (1.0, 13.0, 3.0, 4.0, 3.0, 3.0, 14.0, 1.0, 2.0, 146.0, 'Phone', 'Debit Card', 'Male', 'Single'),
    ('Mobile Phone', 1, 'mid'): (1.0, 12.0, 3.0, 4.0, 3.0, 3.0, 14.0, 1.0, 2.0, 132.0, 'Phone', 'Debit Card', 'Male', 'Married'),
    ('Mobile Phone', 1, 'old'): (1.0, 13.0, 3.0, 4.0, 3.0, 5.0, 15.0, 1.0, 2.0, 148.0, 'Phone', 'Credit Card', 'Male', 'Married'),
    ('Fashion', 0, 'new'): (3.0, 16.0, 3.0, 4.0, 3.0, 3.0, 13.0, 1.0, 2.0, 200.5, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Fashion', 0, 'mid'): (1.0, 14.0, 3.0, 4.0, 3.0, 3.0, 15.0, 2.0, 2.5, 205.0, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Fashion', 0, 'old'): (1.0, 13.0, 3.0, 4.0, 3.0, 5.0, 15.0, 2.0, 2.5, 210.0, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Fashion', 1, 'new'): (2.0, 20.5, 3.0, 4.0, 3.0, 3.0, 15.5, 2.0, 3.5, 216.0, 'Mobile Phone', 'Cash on Delivery', 'Male', 'Married'),
    ('Fashion', 1, 'mid'): (1.0, 16.0, 3.0, 4.0, 3.0, 3.0, 14.0, 1.0, 2.0, 208.5, 'Mobile Phone', 'Debit Card', 'Female', 'Married'),
    ('Fashion', 1, 'old'): (1.0, 13.0, 3.0, 4.0, 3.0, 5.0, 15.0, 1.0, 2.0, 209.0, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Laptop & Accessory', 0, 'new'): (3.0, 15.0, 3.0, 4.0, 3.0, 3.0, 16.0, 1.0, 2.0, 165.0, 'Mobile Phone', 'Debit Card', 'Male', 'Single'),
    ('Laptop & Accessory', 0, 'mid'): (1.0, 14.0, 3.0, 4.0, 3.0, 3.0, 15.0, 1.0, 2.0, 168.0, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Laptop & Accessory', 0, 'old'): (1.0, 14.0, 3.0, 4.0, 3.0, 4.0, 15.0, 1.0, 2.0, 167.0, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Laptop & Accessory', 1, 'new'): (3.0, 16.0, 3.0, 4.0, 3.0, 3.0, 14.0, 1.0, 2.0, 167.0, 'Mobile Phone', 'E wallet', 'Male', 'Married'),
    ('Laptop & Accessory', 1, 'mid'): (1.0, 15.0, 3.0, 4.0, 3.0, 3.0, 14.0, 1.0, 2.0, 168.0, 'Mobile Phone', 'Debit Card', 'Female', 'Married'),
    ('Laptop & Accessory', 1, 'old'): (1.0, 14.0, 3.0, 4.0, 3.0, 4.0, 15.5, 1.0, 2.0, 166.0, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Grocery', 0, 'new'): (1.0, 10.0, 3.0, 4.0, 3.0, 5.0, 15.0, 2.0, 4.0, 270.0, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Grocery', 0, 'mid'): (1.0, 15.0, 3.0, 4.0, 3.0, 5.0, 12.0, 5.0, 6.0, 262.0, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Grocery', 0, 'old'): (1.0, 10.0, 3.0, 4.0, 3.0, 4.0, 15.0, 2.0, 4.0, 271.0, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Grocery', 1, 'new'): (1.0, 18.0, 3.0, 4.0, 3.0, 4.0, 14.0, 2.0, 3.0, 263.5, 'Mobile Phone', 'Credit Card', 'Female', 'Married'),
    ('Grocery', 1, 'mid'): (1.0, 18.0, 3.0, 4.0, 3.0, 4.0, 14.0, 2.0, 3.0, 263.5, 'Mobile Phone', 'Credit Card', 'Female', 'Married'),
    ('Grocery', 1, 'old'): (1.0, 15.0, 3.0, 4.0, 3.0, 4.0, 14.0, 2.0, 3.0, 264.0, 'Mobile Phone', 'Credit Card', 'Female', 'Married'),
    ('Others', 0, 'new'): (1.0, 11.0, 3.0, 4.0, 3.0, 4.0, 14.0, 2.0, 4.0, 302.5, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Others', 0, 'mid'): (1.0, 11.0, 3.0, 4.0, 3.0, 4.0, 14.0, 2.0, 4.0, 302.5, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Others', 0, 'old'): (1.0, 11.0, 3.0, 4.0, 3.0, 4.0, 14.0, 2.0, 4.0, 305.5, 'Mobile Phone', 'Debit Card', 'Male', 'Married'),
    ('Others', 1, 'new'): (1.0, 9.0, 3.0, 4.0, 3.5, 5.0, 14.0, 2.0, 2.0, 303.0, 'Mobile Phone', 'Credit Card', 'Male', 'Married'),
    ('Others', 1, 'mid'): (1.0, 9.0, 3.0, 4.0, 3.5, 5.0, 14.0, 2.0, 2.0, 303.0, 'Mobile Phone', 'Credit Card', 'Male', 'Married'),
    ('Others', 1, 'old'): (1.0, 9.0, 3.0, 4.0, 3.0, 5.0, 14.0, 2.0, 2.0, 309.5, 'Mobile Phone', 'Credit Card', 'Male', 'Married'),
}


def tenure_bucket(months: int) -> str:
    return "new" if months <= 1 else "mid" if months <= 12 else "old"


def segment_profile(category_raw: str, complain: int, tenure_months: int) -> dict:
    """Raw values of the fields the visitor does not enter (see SEGMENT_PROFILES)."""
    return dict(zip(PROFILE_FIELDS, SEGMENT_PROFILES[(category_raw, int(complain), tenure_bucket(int(tenure_months)))]))


def input_scenario(tenure_months: int, complain: int, days: int) -> str:
    """Fallback when none of the engine's top causes is actionable: decide from the visible inputs."""
    if complain:
        return "complaint"
    if tenure_months <= 6:
        return "newbie"
    if days >= 20:
        return "gap"
    return "general"


def apply_preset(number: int) -> None:
    """Quick scenario button: fills the four visible inputs (runs as a callback, before the widgets are drawn)."""
    st.session_state.update(LAB_PRESETS[number]["values"])


LAB_OFFERS = {  # coupon differentiation by risk level (Tab 2 lab)
    "red": {"discount": 10, "code": "VIP-CARE10", "tag": "تخفیف ویژه دلجویی", "css": "gold"},
    "amber": {"discount": 5, "code": "RETURN5", "tag": "تخفیف یادآوری", "css": "silver"},
}
LOGIN_FA = {"Mobile Phone": "موبایل", "Phone": "موبایل", "Computer": "رایانه"}


def build_offer_sms(scenario: str, store: str, coupon: str, discount: int) -> str:
    """Return message; the discount comes from the risk level, the wording from the main cause."""
    d = fa(discount)

    if scenario == "complaint":
        return (
        f"مشتری گرامی، بابت تجربه‌ای که داشتید از طرف {store} صمیمانه عذرخواهی می‌کنیم. \n"
        f"رضایت شما برای ما اهمیت دارد و خوشحال می‌شویم فرصت دوباره‌ای برای جبران این تجربه داشته باشیم.\n"
        f"برای خرید بعدی، {d}٪ تخفیف ویژه برای شما در نظر گرفته‌ایم.\n"
        f"کافی است هنگام ثبت سفارش، کد تخفیف زیر را وارد کنید و از پیشنهاد ویژه خود استفاده کنید.\n"
        f"کد تخفیف: {coupon}\n"
        f"منتظر دیدار دوباره شما هستیم. "
    )

    if scenario == "newbie":
        return (
        f"خوشحالیم که شما را در جمع مشتریان {store} داریم. \n"
        f"برای اینکه دوباره تجربه خریدی لذت‌بخش داشته باشید، {d}٪ تخفیف ویژه برای خرید بعدی شما در نظر گرفته‌ایم.\n"
        f"کافی است هنگام ثبت سفارش، کد تخفیف زیر را وارد کنید و از پیشنهاد ویژه خود استفاده کنید.\n"
        f"کد تخفیف: {coupon}\n"
        f"منتظر دیدار دوباره شما هستیم. "
    )

    if scenario == "gap":
        return (
        f"مدتی است شما را در {store} ندیده‌ایم و خوشحال می‌شویم دوباره میزبان شما باشیم. \n"
        f"برای بازگشت شما، {d}٪ تخفیف ویژه برای خرید بعدی در نظر گرفته‌ایم.\n"
        f"کافی است هنگام ثبت سفارش، کد تخفیف زیر را وارد کنید و از پیشنهاد ویژه خود استفاده کنید.\n"
        f"کد تخفیف: {coupon}\n"
        f"منتظر دیدار دوباره شما هستیم. "
    )

    return (
    f"از همراهی شما با {store} سپاسگزاریم. \n"
    f"برای اینکه دوباره تجربه خریدی لذت‌بخش داشته باشید، {d}٪ تخفیف ویژه برای خرید بعدی شما در نظر گرفته‌ایم.\n"
    f"کافی است هنگام ثبت سفارش، کد تخفیف زیر را وارد کنید و از پیشنهاد ویژه خود استفاده کنید.\n"
    f"کد تخفیف: {coupon}\n"
    f"منتظر دیدار دوباره شما هستیم. "
)


def toggle_detail(customer_id) -> None:
    """Open / close the full-information area of one radar card. While one is open the radar stays frozen."""
    opened = st.session_state["detail_open"]
    opened.symmetric_difference_update({customer_id})
    if not opened:
        st.session_state["last_scan"] = time.time()  # resume the countdown from zero


def _num(value, suffix: str = "") -> str:
    """Persian number for display; None -> 'ثبت نشده'."""
    if value is None:
        return "ثبت نشده"
    v = float(value)
    txt = fa(int(v)) if v == int(v) else fa(round(v, 1))
    return f"{txt}{suffix}"


def full_info_rows(c: dict) -> list:
    """All 19 real fields of a customer with Persian names and readable values (the churn label is never shown)."""
    f = c.get("features") or {}
    tier, sat, cash = f.get("CityTier"), f.get("SatisfactionScore"), f.get("CashbackAmount")
    return [
        ("شناسه مشتری", fa(c["id"])),
        ("سابقه عضویت", _num(f.get("Tenure"), " ماه")),
        ("دستگاه ورود", LOGIN_FA.get(str(f.get("PreferredLoginDevice")), "ثبت نشده")),
        ("سطح شهر", f"درجه {fa(int(tier))}" if tier is not None else "ثبت نشده"),
        ("فاصله انبار تا منزل", _num(f.get("WarehouseToHome"), " کیلومتر")),
        ("روش پرداخت", pay_fa(f.get("PreferredPaymentMode"))),
        ("جنسیت", VALUE_FA.get(str(f.get("Gender")), "ثبت نشده")),
        ("استفادهٔ روزانه از اپلیکیشن", _num(f.get("HourSpendOnApp"), " ساعت")),
        ("دستگاه‌های متصل به حساب", _num(f.get("NumberOfDeviceRegistered"))),
        ("دستهٔ کالای ترجیحی", cat_fa(f.get("PreferedOrderCat")) if f.get("PreferedOrderCat") else "ثبت نشده"),
        ("امتیاز رضایت", f"{fa(int(sat))} از ۵" if sat is not None else "ثبت نشده"),
        ("وضعیت تأهل", VALUE_FA.get(str(f.get("MaritalStatus")), "ثبت نشده")),
        ("آدرس‌های ثبت‌شده", _num(f.get("NumberOfAddress"))),
        ("شکایت در ماه اخیر", "بله" if int(f.get("Complain") or 0) == 1 else "خیر"),
        ("رشد مبلغ سفارش نسبت به پارسال", _num(f.get("OrderAmountHikeFromlastYear"), "٪")),
        ("کوپن‌های استفاده‌شده", _num(f.get("CouponUsed"))),
        ("کل سفارش‌های موفق", _num(f.get("OrderCount"))),
        ("روزهای گذشته از آخرین خرید", _num(f.get("DaySinceLastOrder"), " روز")),
        ("کل پاداش خرید", cash_toman(cash) if cash is not None else "ثبت نشده"),
    ]


EXTRA_CSS = """
<style>
/* ---- countdown bar of the live radar ---- */
.cd { height:3px; background:#E6ECF8; border-radius:99px; overflow:hidden; margin-top:10px; direction:rtl; }
.cd-fill { display:block; height:3px; width:0; background:#3157B7; border-radius:99px; }
@keyframes cdfill0 { from { width:0; } to { width:100%; } }
@keyframes cdfill1 { from { width:0; } to { width:100%; } }

/* ---- minimal radar card ---- */
.mini2 { display:flex; align-items:center; justify-content:space-between; gap:10px; }
.mini2 .c-name { font-size:14px; font-weight:700; color:#1E3A73; line-height:1.8; }
.bar.green > span { background:#22A06B; }
div[class*="st-key-card"] .stButton > button { padding:.2rem .6rem; font-size:12px; font-weight:500; color:#3157B7;
  background:transparent; border:1px dashed #B7C7EA; box-shadow:none; }
div[class*="st-key-card"] .stButton > button:hover { background:#EAF0FD; border-color:#3157B7; color:#1E3A73; }
.fg { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#D6E1F5; border:1px solid #D6E1F5; border-radius:10px;
  overflow:hidden; margin-top:8px; }
.fg div { background:#FFFFFF; padding:6px 10px; font-size:11px; color:#64748B; }
.fg b { display:block; font-size:13px; font-weight:500; color:#172554; }

/* ---- evaluation form (static st.form) ---- */
.st-key-panel_lab_in [data-testid="stForm"] { border:none; padding:0; background:transparent; }
.st-key-panel_lab_in [data-testid="stWidgetLabel"] p { color:#172554; font-size:13px; }
.st-key-panel_lab_in [data-testid="stFormSubmitButton"] button { width:100%; padding:.8rem 1rem; font-size:15px; font-weight:700;
  color:#FFFFFF; background:#3157B7; border:none; border-radius:12px; box-shadow:0 6px 16px rgba(49,87,183,.28); }
.st-key-panel_lab_in [data-testid="stFormSubmitButton"] button:hover { background:#1E3A73; color:#FFFFFF; }
.st-key-presets_zone .stButton > button { padding:.15rem .3rem; font-size:12px; font-weight:500; border-radius:8px; min-height:30px;
  color:#1E3A73; background:#F4F7FE; border:1px solid #D6E1F5; }
.st-key-presets_zone .stButton > button:hover { border-color:#3157B7; color:#3157B7; background:#EAF0FD; }
.helpcard { padding:18px 16px; border-radius:12px; background:#F4F7FE; border:1px dashed #B7C7EA; color:#1E3A73;
  font-size:13px; line-height:2; text-align:center; }

/* ---- clean SMS panel ---- */
.sms2 { background:#FFFFFF; border:1px solid #D6E1F5; border-radius:14px; overflow:hidden; box-shadow:0 2px 12px rgba(30,58,115,.06); }
.sms2-head { display:flex; align-items:center; justify-content:space-between; gap:8px; padding:10px 14px; border-bottom:1px solid #D6E1F5; }
.sms2-head b { font-size:14px; color:#1E3A73; }
.tier { padding:2px 10px; border-radius:99px; font-size:11px; font-weight:700; }
.tier.gold { color:#6B4E00; background:#FFF1C4; border:1px solid #D9B648; }
.tier.silver { color:#3B4656; background:#EEF1F5; border:1px solid #AEB7C4; }
.code-row { display:flex; align-items:center; gap:10px; padding:12px 14px 0; }
.cbox { display:inline-flex; align-items:baseline; gap:8px; padding:6px 14px; border-radius:10px; font-size:12px; }
.cbox b { font-size:17px; letter-spacing:1px; direction:ltr; display:inline-block; }
.cbox.gold { color:#6B4E00; background:linear-gradient(135deg,#FFF3CF,#F5DA86); border:1px solid #D4AF37; }
.cbox.silver { color:#3B4656; background:linear-gradient(135deg,#F7F9FB,#D9DFE7); border:1px solid #AEB7C4; }
.pct { font-size:13px; font-weight:700; color:#1E3A73; }
.sms2-body { margin:10px 14px 12px; padding:12px 14px; border-radius:10px; background:#F8FAFE; border-right:3px solid #3157B7;
  font-size:13px; line-height:2; white-space:pre-line; color:#172554; }
.sms2-ok { margin:0 14px 12px; padding:8px 12px; border-radius:8px; background:#E6F6EE; color:#14532D; font-size:13px; font-weight:700; }
.lab-note { font-size:11px; color:#64748B; line-height:1.8; margin-top:4px; }
.mini-title { font-size:12px; color:#64748B; margin:10px 0 4px; }
</style>
"""


# --------------------------------------------------------------------------- #
# Global CSS (plain string). Fonts are applied to specific selectors (never "*") so that
# Streamlit's icon font is not overridden. Secondary text (#64748B) is used on white cards only;
# text placed directly on the blue background uses darker colours for accessible contrast.
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&display=swap');

:root { --bg:#C4D5F9; --blue:#3157B7; --navy:#1E3A73; --ink:#172554; --muted:#64748B; --teal:#2AAFA3; --border:#D6E1F5;
        --low:#22A06B; --mid:#E6A23C; --high:#D9534F; --on-bg:#334155; }

html, body, .stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li, .stApp [data-baseweb],
.stApp [data-testid="stMarkdownContainer"] *, .stApp [data-testid="stExpander"] summary * {
  font-family:'Vazirmatn','Tahoma','Segoe UI',sans-serif;
}
.stApp { background:linear-gradient(180deg,#C4D5F9 0%,#D3E0FB 100%); background-attachment:fixed; color:var(--ink);
         direction:rtl; text-align:right; font-size:13px; }
.stApp p, .stApp label, .stApp li, .stApp [data-testid="stWidgetLabel"] p, .stApp [data-baseweb="select"] * { font-size:13px; }
.block-container { padding-top:1rem; padding-bottom:2rem; max-width:1320px; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility:hidden; height:0; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display:none !important; }
p, label, .stMarkdown { direction:rtl; text-align:right; }
[data-testid="stVerticalBlock"] { gap:.6rem; }

/* ---------- brand header ---------- */
.brand { background:linear-gradient(135deg,var(--navy),#27488F); border-radius:16px; padding:16px 22px; box-shadow:0 4px 18px rgba(30,58,115,.22); }
.brand h1 { margin:0; padding:0; font-size:21px; font-weight:700; line-height:1.6; color:#FFFFFF; }
.brand p { margin:2px 0 0; font-size:13px; color:#D6E1F5; }

/* ---------- surfaces ---------- */
.card, div[class*="st-key-card"], div[class*="st-key-panel"] {
  background:#FFFFFF; border:1px solid var(--border); border-radius:16px; box-shadow:0 2px 12px rgba(30,58,115,.06);
}
.card { padding:12px 14px; }
div[class*="st-key-card"], div[class*="st-key-panel"] { padding:14px 16px; }
div[class*="st-key-card_red"] { background:#FEF4F3; border:1px solid #F1C3C1; border-top:4px solid var(--high); }
div[class*="st-key-card_amber"] { background:#FFF9EE; border:1px solid #F4D9A8; border-top:4px solid var(--mid); }
div[class*="st-key-card_green"] { background:#F0FAF5; border:1px solid #B5E0CB; border-top:4px solid var(--low); }
.h2 { font-size:17px; font-weight:700; color:var(--navy); margin:14px 2px 6px; }
.h2 small { color:var(--on-bg); font-weight:400; font-size:13px; margin-inline-start:8px; }
.lbl { color:var(--muted); font-weight:500; font-size:12px; margin:4px 0 2px; }
.tag { display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:99px; font-size:12px; color:#334155;
  background:#F1F5FD; border:1px solid var(--border); }
.tag.blue { color:var(--blue); background:#EAF0FD; border-color:#C5D4F3; }
.tag.teal { color:#14766D; background:#E5F6F4; border-color:#B4E3DE; }
.tags { display:flex; gap:6px; flex-wrap:wrap; }

/* ---------- navigation / segmented choices ---------- */
div[class*="st-key-seg"] div[role="radiogroup"] { gap:4px; padding:3px; border-radius:10px; flex-wrap:wrap; background:#EAF0FD; border:1px solid var(--border); }
div[class*="st-key-seg"] label { flex:1 1 80px; justify-content:center; margin:0; padding:7px 10px; border-radius:8px; cursor:pointer; }
.st-key-seg_nav div[role="radiogroup"] { background:rgba(255,255,255,.55); border:1px solid rgba(255,255,255,.7); }
.st-key-seg_nav label { flex:1 1 300px; padding:10px 14px; }
div[class*="st-key-seg"] label > div:first-child { display:none; }
div[class*="st-key-seg"] label p { margin:0; font-weight:500; color:#334155; text-align:center; font-size:13px; }
div[class*="st-key-seg"] label:has(input:checked) { background:#FFFFFF; box-shadow:0 1px 5px rgba(30,58,115,.18); }
div[class*="st-key-seg"] label:has(input:checked) p { color:var(--blue); font-weight:700; }
.st-key-seg_nav label:has(input:checked) { background:var(--blue); }
.st-key-seg_nav label:has(input:checked) p { color:#FFFFFF; }

/* ---------- KPI ---------- */
.kpi { position:relative; }
.kpi::before { content:""; position:absolute; top:0; right:14px; width:28px; height:3px; border-radius:0 0 3px 3px; background:var(--blue); }
.kpi .lb { color:var(--muted); font-size:12px; margin-bottom:2px; }
.kpi .vl { font-size:20px; font-weight:700; line-height:1.4; color:var(--navy); }
.kpi .vl.teal { color:#14766D; } .kpi .vl.red { color:var(--high); }
.kpi .vl small { font-size:12px; font-weight:500; color:var(--muted); margin-inline-start:5px; }
.kpi .sb { font-size:11px; color:var(--muted); }

/* ---------- analysis boxes ---------- */
.an-title { font-size:17px; font-weight:700; color:var(--navy); margin:0 0 2px; }
.an-kicker { font-size:11px; font-weight:700; color:var(--blue); margin-bottom:2px; }
.an-nums { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0; }
.an-num { padding:6px 12px; border-radius:10px; background:#F4F7FE; border:1px solid var(--border); font-size:12px; color:var(--muted); }
.an-num b { display:block; font-size:17px; color:var(--navy); }
.an-num.r b { color:var(--high); } .an-num.g b { color:var(--low); } .an-num.b b { color:var(--blue); }
.an ul { margin:6px 0; padding:0 18px 0 0; color:#1E293B; font-size:13px; line-height:1.95; }
.an li b { color:var(--ink); }
.takeaway { margin-top:8px; padding:8px 12px; border-radius:10px; border-right:3px solid var(--blue); background:#F1F5FD; color:#1E293B; font-size:13px; line-height:1.9; }
.takeaway b { color:var(--blue); }
.road { min-height:150px; }
.road .st { font-size:11px; font-weight:700; color:var(--blue); }
.road h4 { margin:2px 0 6px; font-size:15px; font-weight:700; color:var(--navy); }
.road ul { margin:0; padding:0 16px 0 0; color:#1E293B; font-size:13px; line-height:1.9; }
.road .kp { margin-top:6px; padding-top:6px; border-top:1px dashed var(--border); font-size:12px; color:var(--muted); }

/* ---------- radar customer cards (CRM style) ---------- */
.c-top { display:flex; align-items:center; gap:10px; }
.avatar { width:38px; height:38px; flex:0 0 38px; border-radius:10px; display:grid; place-items:center; font-weight:700; color:#fff; font-size:12px; }
.av-red { background:var(--high); } .av-amber { background:var(--mid); } .av-green { background:var(--low); }
.c-id { flex:1; min-width:0; }
.c-name { font-size:15px; font-weight:700; color:var(--navy); }
.c-sub { font-size:11px; color:var(--muted); }
.badge { display:inline-block; padding:3px 12px; border-radius:99px; font-size:12px; font-weight:700; color:#fff; white-space:nowrap; }
.badge.red { background:var(--high); } .badge.amber { background:#B7791F; } .badge.green { background:#15803D; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:6px 14px; margin:12px 0 8px; }
.grid .it { font-size:12px; color:var(--muted); }
.grid .it b { display:block; font-size:13px; color:var(--ink); font-weight:500; }
.bar { height:6px; border-radius:99px; background:#E6ECF8; margin:8px 0 6px; overflow:hidden; }
.bar > span { display:block; height:100%; border-radius:99px; }
.bar.red > span { background:var(--high); } .bar.amber > span { background:var(--mid); }
.riskline { display:flex; align-items:center; gap:8px; font-weight:700; font-size:15px; }
.riskline.red { color:#B9403C; } .riskline.amber { color:#A8710F; }
.cause { display:block; padding:7px 10px; border-radius:8px; font-size:12px; margin-top:6px; color:var(--ink); }
.cause.red { background:#FBE6E5; } .cause.amber { background:#FCF0D6; }
.stable { display:flex; align-items:center; gap:8px; padding:10px 12px; border-radius:10px; font-weight:700; font-size:13px; color:#14532D; background:#DDF3E8; border:1px solid #B5E0CB; margin-top:6px; }

/* ---------- lab: value card + draft panel ---------- */
.riskbox { padding:12px 14px; border-radius:12px; border:1px solid var(--border); background:#FFFFFF; }
.riskbox.red { background:#FEF4F3; border-color:#F1C3C1; } .riskbox.amber { background:#FFF9EE; border-color:#F4D9A8; }
.value-card { text-align:center; padding:14px 12px; border-radius:12px; border:1px solid #B4E3DE; background:linear-gradient(160deg,#ECF9F7,#FFFFFF); }
.value-card .cap { color:#14766D; font-weight:700; font-size:13px; }
.value-card .big { font-size:28px; font-weight:700; line-height:1.6; color:var(--navy); }
.value-card .big small { font-size:13px; font-weight:500; color:var(--muted); margin-inline-start:6px; }
.value-card .note { font-size:11px; color:var(--muted); line-height:1.7; }
.draft { background:#FFFFFF; border:1px solid var(--border); border-radius:14px; overflow:hidden; box-shadow:0 2px 12px rgba(30,58,115,.06); }
.draft-head { padding:9px 14px; background:var(--navy); color:#FFFFFF; font-weight:700; font-size:14px; }
.draft-meta { display:flex; gap:8px; flex-wrap:wrap; padding:10px 14px 0; }
.draft-meta .m { padding:4px 10px; border-radius:8px; font-size:12px; background:#F4F7FE; border:1px solid var(--border); color:var(--muted); }
.draft-meta .m b { color:var(--ink); font-weight:700; }
.draft-meta .m.ok b { color:#14766D; }
.draft-body { margin:10px 14px 4px; padding:12px 14px; border-radius:10px; background:#F8FAFE; border-right:3px solid var(--blue);
  font-size:13px; line-height:2; white-space:pre-line; color:var(--ink); }
.coupon { display:inline-block; direction:ltr; padding:0 9px; border-radius:6px; font-weight:700; letter-spacing:1px; color:#fff; background:var(--navy); }
.draft-foot { padding:0 14px 12px; font-size:11px; color:var(--muted); }

/* ---------- native widgets ---------- */
.stButton > button { width:100%; border-radius:10px; padding:.45rem .9rem; font-weight:500; font-size:13px; color:var(--ink); background:#FFFFFF; border:1px solid #B7C7EA; }
.stButton > button:hover { border-color:var(--blue); color:var(--blue); }
.stButton > button:disabled { color:#14766D; border-color:#B4E3DE; background:#E5F6F4; }
.st-key-register_zone .stButton > button { padding:.75rem 1rem; font-size:14px; font-weight:700; color:#fff; border:none; border-radius:12px; background:var(--blue); box-shadow:0 6px 16px rgba(49,87,183,.25); }
.st-key-register_zone .stButton > button:hover { background:var(--navy); color:#fff; }
.st-key-register_zone .stButton > button:disabled { background:#E5F6F4; color:#14766D; box-shadow:none; }
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"] { background:#FFFFFF !important; border-color:#B7C7EA !important; border-radius:10px !important; min-height:36px; }
[data-testid="stSlider"] [role="slider"] { background:var(--blue) !important; border:2px solid #fff !important; box-shadow:0 1px 5px rgba(30,58,115,.35) !important; }
[data-testid="stSliderThumbValue"] { color:var(--navy) !important; font-size:12px !important; }
.foot { text-align:center; color:var(--on-bg); font-size:11px; margin-top:14px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================================================================== #
# LOGIC (no Streamlit rendering below this line until the "UI" marker)
# =========================================================================== #
_FA_DIGITS = str.maketrans("0123456789,.%", "۰۱۲۳۴۵۶۷۸۹٬٫٪")


def fa(value) -> str:
    """Convert digits (and , . %) to Persian typography."""
    return str(value).translate(_FA_DIGITS)


def fmt_int(value: float) -> str:
    sign = "−" if value < 0 else ""
    return sign + fa(f"{abs(int(round(value))):,}")


def fmt_toman(value: float) -> tuple[str, str]:
    """Return (number, unit) such as ('۱۱٫۴', 'میلیارد تومان')."""
    if value >= 1e9:
        return fa(f"{value / 1e9:.1f}"), "میلیارد تومان"
    if value >= 1e6:
        return fa(f"{value / 1e6:,.0f}"), "میلیون تومان"
    return fmt_int(value), "تومان"


def cash_toman(raw_value: float) -> str:
    """Display-only conversion of a raw cashback value: display_toman = raw_value * 2000."""
    return f"{fmt_int(float(raw_value) * 2000)} تومان"


def pay_fa(raw) -> str:
    """Display-only Persian label for PreferredPaymentMode (the raw value is what the engine receives)."""
    if raw is None:
        return "نامشخص"
    return PAYMENT_FA.get(str(raw).strip(), "سایر")


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def compact(markup: str) -> str:
    """Strip indentation/newlines of an HTML template so Markdown never sees an indented code block."""
    return "".join(line.strip() for line in markup.splitlines())


def tone_of(risk: float) -> str:
    if risk >= RED_FROM:
        return "red"
    if risk >= AMBER_FROM:
        return "amber"
    return "green"


def cat_fa(raw) -> str:
    return CATEGORY_FA.get(str(raw).strip(), str(raw))


def _clean(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def row_to_features(row: pd.Series) -> dict:
    """RAW customer record exactly as the engine expects it (no display conversion)."""
    skip = {"CustomerID", "Churn", "category_fa"}
    return {key: _clean(val) for key, val in row.items() if key not in skip}


def describe_driver(driver: dict) -> dict:
    """Turn a raw driver from predict_single_customer into short Persian display text."""
    feature, value = str(driver["feature"]), driver["value"]
    num = value if isinstance(value, (int, float)) else 0

    if " != " in feature or " = " in feature:
        sep = " != " if " != " in feature else " = "
        col, val = feature.split(sep, 1)
        shown = pay_fa(val) if col == "PreferredPaymentMode" else VALUE_FA.get(val, val)
        label = f"{COLUMN_FA.get(col, col)}: {shown}"
        return {"title": label if sep == " = " else f"غیر از «{label}»", "detail": "الگوی رفتاری پرریسک"}

    table = {
        "Tenure": ("سابقهٔ کم همراهی", f"{fa(int(num))} ماه همراهی"),
        "Complain": ("ثبت شکایت", "شکایت بدون پاسخ رضایت‌بخش" if num >= 1 else "بدون شکایت"),
        "WarehouseToHome": ("فاصلهٔ زیاد تا انبار", f"حدود {fa(int(num))} کیلومتر"),
        "DaySinceLastOrder": ("فاصله از آخرین خرید", f"{fa(int(num))} روز از آخرین سفارش"),
        "avg_cashbk_per_order": ("میانگین بازگشت وجه هر سفارش", cash_toman(num)),
        "CashbackAmount": ("بازگشت وجه", cash_toman(num)),
        "SatisfactionScore": ("الگوی رضایت", f"امتیاز {fa(int(num))} از ۵"),
        "OrderCount": ("تعداد سفارش اندک", f"{fa(int(num))} سفارش"),
        "CouponUsed": ("استفادهٔ کم از کوپن", f"{fa(int(num))} کوپن"),
        "HourSpendOnApp": ("زمان کم در اپلیکیشن", f"حدود {fa(int(num))} ساعت"),
        "NumberOfDeviceRegistered": ("تعداد دستگاه ثبت‌شده", f"{fa(int(num))} دستگاه"),
        "NumberOfAddress": ("تعداد آدرس ثبت‌شده", f"{fa(int(num))} آدرس"),
        "CityTier": ("سطح شهر", f"سطح {fa(int(num))}"),
        "OrderAmountHikeFromlastYear": ("تغییر مبلغ سفارش", f"{fa(int(num))}٪ نسبت به پارسال"),
    }
    title, detail = table.get(feature, ("عامل رفتاری", "الگوی مشابه مشتریان ریزشی"))
    return {"title": title, "detail": detail}


# ---- return scenarios (chosen from the main risk cause) ----
SCENARIOS = {
    "complaint": {"label": "پیگیری و جبران شکایت", "prefix": "JOBRAN", "discount": 30},
    "newbie": {"label": "خوش‌آمدگویی و مشوق خرید دوم", "prefix": "KHOSHAMAD", "discount": 15},
    "gap": {"label": "دعوت به بازگشت با ارسال رایگان", "prefix": "BAZGASHT", "discount": 0},
    "general": {"label": "پیشنهاد بازگشت مشتری", "prefix": "VAFADARI", "discount": 10},
}


def pick_scenario(drivers: list[dict]) -> tuple[str, dict | None]:
    """First driver (by impact) that matches an actionable cause AND is consistent with its value."""
    for d in drivers:
        f, v = d["feature"], d["value"]
        if f == "Complain" and v >= 1:
            return "complaint", d
        if f == "Tenure" and v <= 12:
            return "newbie", d
        if f == "DaySinceLastOrder" and v >= 5:
            return "gap", d
    return "general", None


def main_cause(customer: dict) -> dict | None:
    """The actionable cause when there is one, otherwise the strongest driver."""
    if customer.get("trigger"):
        return customer["trigger"]
    return customer["drivers"][0] if customer["drivers"] else None


def build_sms(scenario: str, store: str, coupon: str) -> str:
    d = fa(SCENARIOS[scenario]["discount"])
    if scenario == "complaint":
        return (f"مشتری گرامی، بابت مشکل پیش‌آمده از طرف {store} صمیمانه عذرخواهی می‌کنیم. "
                f"واحد پشتیبانی پیگیر پروندهٔ شماست.\nبه‌عنوان جبران، کد {d}٪ تخفیف برای خرید بعدی شما فعال شد:\n{coupon}\nاعتبار: ۷۲ ساعت")
    if scenario == "newbie":
        return f"به خانوادهٔ {store} خوش آمدید. برای خرید دوم، {d}٪ تخفیف ویژهٔ شما:\n{coupon}\nاعتبار: ۷ روز"
    if scenario == "gap":
        return (f"مدتی است شما را در {store} ندیده‌ایم. سفارش بعدی شما با ارسال رایگان تحویل می‌شود.\n"
                f"کد بازگشت:\n{coupon}\nاعتبار: ۱۰ روز")
    return f"از همراهی شما با {store} سپاسگزاریم. برای بازگشت، {d}٪ تخفیف ویژه:\n{coupon}\nاعتبار: ۷ روز"


def retention_value(category_raw: str, risk: float) -> int:
    """Illustrative estimate = assumed annual category value x churn probability (rounded to 10,000 Toman)."""
    return int(round(CATEGORY_ANNUAL_VALUE.get(category_raw, 3_500_000) * risk / 100 / 10_000) * 10_000)


class Engine:
    """Loads churn_inference + the stream once (st.cache_resource) and scores customers on demand."""

    def __init__(self) -> None:
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        import churn_inference  # noqa: WPS433

        self._predict = churn_inference.predict_single_customer
        df = pd.read_csv(STREAM_PATH)
        df["CustomerID"] = df["CustomerID"].astype(int)
        df["category_fa"] = df["PreferedOrderCat"].map(cat_fa)
        self.df = df
        self.by_id = {int(row["CustomerID"]): row for _, row in df.iterrows()}
        self.scores: dict = {}
        self.lab_cache: dict = {}
        self._predict(row_to_features(df.iloc[0]), model_path=MODEL_PATH)  # warm-up

    @staticmethod
    def _package(pid, result: dict, features: dict, category: str) -> dict:
        drivers = [{**d, **describe_driver(d)} for d in result["top_risk_drivers"]]
        scenario, trigger = pick_scenario(drivers)
        return {"id": pid, "risk": float(result["churn_probability_pct"]), "drivers": drivers, "category": category,
                "tenure": features.get("Tenure"), "payment": features.get("PreferredPaymentMode"),
                "complain": int(features.get("Complain") or 0), "features": features,
                "scenario": scenario, "trigger": trigger}

    def score(self, customer_id: int) -> dict:
        cached = self.scores.get(customer_id)
        if cached is None:
            row = self.by_id[customer_id]
            features = row_to_features(row)
            result = self._predict(features, model_path=MODEL_PATH)
            cached = self._package(int(customer_id), result, features, str(row["category_fa"]))
            self.scores[customer_id] = cached
        return cached

    def lab_score(self, tenure_months: int, complain: int, category_raw: str, days: int,
                  city_tier: int, satisfaction: int, orders: int) -> dict:
        """Score a customer built from the seven form factors (other fields: medians of similar historical customers)."""
        key = (tenure_months, complain, category_raw, days, city_tier, satisfaction, orders)
        if key not in self.lab_cache:
            features = segment_profile(category_raw, complain, tenure_months)
            features.update({"Tenure": float(tenure_months), "Complain": int(complain), "PreferedOrderCat": category_raw,
                             "DaySinceLastOrder": float(days), "CityTier": float(city_tier),
                             "SatisfactionScore": float(satisfaction), "OrderCount": float(orders)})
            result = self._predict(features, model_path=MODEL_PATH)
            self.lab_cache[key] = self._package("LAB", result, features, cat_fa(category_raw))
        return self.lab_cache[key]


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    return Engine()


def do_scan() -> None:
    """Load exactly two fresh random customers from the stream and score them with the engine."""
    engine = get_engine()
    rng = np.random.default_rng()
    previous = {c["id"] for c in st.session_state.get("radar", [])}
    pool = engine.df[~engine.df["CustomerID"].isin(previous)]
    if len(pool) < RADAR_SIZE:
        pool = engine.df
    scored: list = []
    for _attempt in range(20):  # bounded retries: the loop can never hang
        if len(scored) >= RADAR_SIZE:
            break
        ids = rng.choice(pool["CustomerID"].to_numpy(), size=RADAR_SIZE, replace=False).tolist()
        for cid in ids:
            try:
                customer = engine.score(int(cid))
            except Exception:  # one bad record must never break the live demo
                continue
            if customer["id"] not in {c["id"] for c in scored} and len(scored) < RADAR_SIZE:
                scored.append(customer)
    st.session_state["radar"] = scored
    st.session_state["scan_count"] = st.session_state.get("scan_count", 0) + 1
    st.session_state["last_scan"] = time.time()


def register_scenario(lab_id: str) -> None:
    """Demo only: records the return scenario. Nothing is actually sent."""
    st.session_state["registered"].add(lab_id)
    st.session_state["flash"] = lab_id


def reset_demo() -> None:
    st.session_state.update({
        "registered": set(), "flash": None, "scan_count": 0, "radar": [], "store_name": DEFAULT_STORE, "view": VIEW_BI,
        "auto_running": True, "detail_open": set(), "lab_eval": None, **LAB_DEFAULTS,
    })
    do_scan()


# ------------------------------- Plotly (white, minimal) --------------------- #
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

GRID = "rgba(214, 225, 245, 0.4)"

# Real counts behind the satisfaction chart: score -> (loyal customers, churned customers).
# They add up to 4,682 loyal + 948 churned and reproduce the churn rates stored in DATASET["satisfaction"].
SATISFACTION_COUNTS = {1: (1030, 134), 2: (512, 74), 3: (1406, 292), 4: (890, 184), 5: (844, 264)}

# Chart-only category names and colours (highest risk first).
CHART_CATEGORY_LABELS = {"Mobile Phone": "تلفن همراه", "Fashion": "مد و پوشاک", "Laptop & Accessory": "لپ‌تاپ و کامپیتور",
                         "Others": "سایر", "Grocery": "سوپرمارکت و خواربار"}
CHART_CATEGORY_COLORS = {"Mobile Phone": HIGH, "Fashion": MID, "Laptop & Accessory": BLUE, "Others": BLUE, "Grocery": LOW}


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    """Shared look: white surface, faint grid, compact margins, 11-12 px Persian font."""
    fig.update_layout(
        height=height, autosize=True, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"family": "Vazirmatn, Tahoma, sans-serif", "color": INK, "size": 12},
        margin={"l": 10, "r": 10, "t": 25, "b": 20}, showlegend=False,
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": BORDER, "font": {"family": "Vazirmatn, Tahoma", "color": INK, "size": 12}},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor=BORDER, automargin=True,
                     tickfont={"color": MUTED, "size": 11})
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, showline=False, automargin=True,
                     tickfont={"color": MUTED, "size": 11})
    return fig


def _round_bars(fig: go.Figure, radius: int = 6) -> go.Figure:
    """Rounded bar corners (Plotly >= 5.24); silently skipped on older versions."""
    try:
        fig.update_traces(marker_cornerradius=radius, selector={"type": "bar"})
    except ValueError:
        pass
    return fig


def _area_line(x: list, y: list, **kwargs) -> go.Scatter:
    """Smooth brand-blue line whose area is filled with a very soft transparent gradient (plain fill on old Plotly)."""
    try:
        return go.Scatter(x=x, y=y, fill="tozeroy",
                          fillgradient={"type": "vertical", "colorscale": [[0, "rgba(49, 87, 183, 0.0)"], [1, "rgba(49, 87, 183, 0.15)"]]},
                          **kwargs)
    except ValueError:
        return go.Scatter(x=x, y=y, fill="tozeroy", fillcolor="rgba(49, 87, 183, 0.15)", **kwargs)


# 1) Satisfaction paradox: loyal vs churned volume per score + churn-rate line
def fig_satisfaction() -> go.Figure:
    scores = list(SATISFACTION_COUNTS)
    labels = [f"امتیاز {fa(s)}" for s in scores]
    loyal = [SATISFACTION_COUNTS[s][0] for s in scores]
    churned = [SATISFACTION_COUNTS[s][1] for s in scores]
    rates = [DATASET["satisfaction"][s] for s in scores]
    tips = [f"امتیاز {fa(s)}: وفادار {fmt_int(l)} نفر | ریزش {fmt_int(c)} نفر (نرخ {fa(r)}٪)"
            for s, l, c, r in zip(scores, loyal, churned, rates)]
    rate_text = [f"<b>{fa(r)}٪</b>" if s in (1, 5) else f"{fa(r)}٪" for s, r in zip(scores, rates)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name="مشتریان وفادار", x=labels, y=loyal, marker={"color": NAVY, "line": {"width": 0}},
                         customdata=tips, hovertemplate="%{customdata}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Bar(name="مشتریان ریزش‌کرده", x=labels, y=churned, marker={"color": HIGH, "line": {"width": 0}},
                         customdata=tips, hovertemplate="%{customdata}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(name="نرخ ریزش", x=labels, y=rates, mode="lines+markers+text", text=rate_text, textposition="top center",
                             cliponaxis=False, line={"color": MID, "width": 3, "shape": "spline"}, textfont={"size": 11, "color": INK},
                             marker={"size": [9, 9, 9, 9, 13], "color": [MID] * 4 + [HIGH], "line": {"color": "#FFFFFF", "width": 2}},
                             customdata=tips, hovertemplate="%{customdata}<extra></extra>"), secondary_y=True)
    style_fig(fig, 260)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06, showlegend=True,
                      legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.16, "font": {"size": 11, "color": MUTED}},
                      margin={"l": 10, "r": 10, "t": 38, "b": 20})
    fig.update_yaxes(range=[0, max(loyal) * 1.65], showticklabels=False, showgrid=True, secondary_y=False)
    fig.update_yaxes(range=[-45, 30], showticklabels=False, showgrid=False, secondary_y=True)  # keeps the rate line above the bars
    return _round_bars(fig, 5)


# 2) Complaint lever: churn rate with vs without complaint
def fig_complaint() -> go.Figure:
    labels = list(DATASET["complain"].keys())            # ["بدون شکایت", "دارای شکایت"]
    values = list(DATASET["complain"].values())          # [10.9, 31.7]
    fig = go.Figure(go.Bar(
        x=labels, y=values, width=0.36, marker={"color": [TEAL, HIGH], "line": {"width": 0}},
        text=[f"<b>{fa(v)}٪</b>" for v in values], textposition="outside", cliponaxis=False, textfont={"size": 13, "color": INK},
        customdata=[f"{l}: نرخ ریزش {fa(v)}٪" for l, v in zip(labels, values)], hovertemplate="%{customdata}<extra></extra>"))
    fig.add_annotation(x=0.5, xref="paper", y=values[1] * 1.2, yref="y", showarrow=False,
                       text=f"<b>شکاف ≈ {fa(round(values[1] / values[0], 1))} برابر</b>", font={"size": 13, "color": HIGH})
    style_fig(fig, 290)
    style_fig(fig, 290)

    fig.update_layout(
    title=dict(
        text="مقایسه ریزش مشتریان با و بدون شکایت",
        x=0.5,
        xanchor="center",
        font=dict(
            size=12,
            color=MUTED,
            family="Vazirmatn, Tahoma, sans-serif"
        )
    ),
    margin=dict(t=45, b=45, l=6, r=6)
)
    fig.update_yaxes(range=[0, values[1] * 1.45], showticklabels=False)

    fig.update_xaxes(
    showticklabels=True,
    tickfont={"size": 11, "color": MUTED},
    showline=False,
    ticks=""
)
    return _round_bars(fig, 8)


# 3) First-month cliff: retention by tenure as a soft spline area
def fig_tenure() -> go.Figure:
    labels = ["تا ۱ ماه", "تا ۳ ماه", "تا ۶ ماه", "تا ۱۲ ماه", "بیش از ۱۲ ماه"]

    churn = list(DATASET["tenure"].values())
    retention = [round(100 - v, 1) for v in churn]

    tips = [
        f"سابقهٔ {l}: ماندگاری {fa(r)}٪ | ریزش {fa(c)}٪"
        for l, r, c in zip(labels, retention, churn)
    ]

    last = len(labels) - 1

    fig = go.Figure()

    fig.add_trace(_area_line(
        x=labels,
        y=retention,
        mode="lines+markers+text",
        text=[f"{fa(r)}٪" for r in retention],
        cliponaxis=False,
        textposition=["bottom center"] + ["top center"] * last,
        textfont={"size": 11, "color": INK},
        line={
            "color": BLUE,
            "width": 3,
            "shape": "spline"
        },
        marker={
            "size": [13] + [9] * last,
            "color": [HIGH] + [BLUE] * last,
            "line": {
                "color": "#FFFFFF",
                "width": 2.5
            }
        },
        customdata=tips,
        hovertemplate="%{customdata}<extra></extra>"
    ))

    fig.add_annotation(
        x=labels[0],
        y=retention[0],
        yshift=-34,
        showarrow=False,
        font={"size": 11, "color": HIGH},
        text=f"<b>ریزش {fa(churn[0])}٪ در ماه اول</b>"
    )

    style_fig(fig, 310)

    fig.update_layout(
        title=dict(
            text="نرخ ماندگاری مشتری بر اساس سابقه",
            x=0.5,
            xanchor="center",
            font=dict(
                size=12,
                color=MUTED
            )
        ),
        margin=dict(
            t=45,
            b=45,
            l=10,
            r=30
        )
    )

    fig.update_yaxes(
        range=[25, 112],
        ticksuffix="٪",
        showticklabels=False
    )

    return fig


# 4) Category risk: thin horizontal bars, highest risk first
def fig_categories() -> go.Figure:
    items = sorted(DATASET["category"].items(), key=lambda kv: kv[1], reverse=True)
    labels = [CHART_CATEGORY_LABELS.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    colors = [CHART_CATEGORY_COLORS.get(k, BLUE) for k, _ in items]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", width=0.3, marker={"color": colors, "line": {"width": 0}},
        text=[f"{fa(v)}٪" for v in values], textposition="outside", cliponaxis=False, textfont={"size": 12, "color": INK},
        customdata=[f"{l}: نرخ ریزش {fa(v)}٪" for l, v in zip(labels, values)], hovertemplate="%{customdata}<extra></extra>"))
    style_fig(fig, 260)
    fig.update_layout(margin=dict(l=100, r=40, t=15, b=15))
    fig.update_xaxes(range=[0, max(values) * 1.3], showticklabels=False, showgrid=False, showline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False, ticks="", tickfont={"size": 12, "color": INK})
    return _round_bars(fig, 4)


# 5) Cashback and warehouse distance: two sub-plots
def fig_behaviour() -> go.Figure:
    rates, distance = DATASET["cashback_rate"], DATASET["distance"]

    # برای نمودار فاصله ارسال
    dist_labels = list(distance.keys())
    dist_rates = list(distance.values())

    # برای نمودار پاداش خرید
    # از برچسب‌های پولی استفاده نمی‌کنیم تا بازه‌های مبلغی روی محور X نمایش داده نشوند.
    # خود تابع cashback_band_labels پایین‌تر همچنان برای بخش‌های دیگر برنامه باقی می‌ماند.
    cash_x = list(range(len(rates)))

    # Plotly columns:
    # col=1 -> سمت چپ: فاصله ارسال
    # col=2 -> سمت راست: پاداش خرید
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            "نرخ ریزش بر اساس فاصله ارسال",
            "نرخ ریزش در سطوح مختلف پاداش خرید"
        )
    )

    # ─────────────────────────────────────────
    # نمودار فاصله ارسال
    # ─────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dist_labels,
            y=dist_rates,
            mode="lines+markers+text",
            text=[f"{fa(v)}٪" for v in dist_rates],
            textposition="top center",
            cliponaxis=False,

            textfont={
                "size": 11,
                "color": INK
            },

            line={
                "color": BLUE,
                "width": 3,
                "shape": "linear"
            },

            marker={
                "size": 11,
                "color": [LOW, MID, HIGH],
                "line": {
                    "color": "#FFFFFF",
                    "width": 2
                }
            },

            customdata=[
                f"فاصله ارسال {l}: نرخ ریزش {fa(v)}٪"
                for l, v in zip(dist_labels, dist_rates)
            ],

            hovertemplate="%{customdata}<extra></extra>"
        ),
        row=1,
        col=1
    )

    # ─────────────────────────────────────────
    # نمودار پاداش خرید
    # ─────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=cash_x,
            y=rates,
            mode="lines+markers+text",
            text=[f"{fa(v)}٪" for v in rates],
            textposition="top center",
            cliponaxis=False,

            textfont={
                "size": 11,
                "color": INK
            },

            line={
                "color": BLUE,
                "width": 3,
                "shape": "spline"
            },

            marker={
                "size": 11,
                "color": [HIGH, MID, LOW, LOW],
                "line": {
                    "color": "#FFFFFF",
                    "width": 2
                }
            },

            # دیگر مبلغ یا بازه پولی در Hover نمایش داده نمی‌شود
            customdata=[
                f"سطح پاداش خرید {i + 1}: نرخ ریزش {fa(v)}٪"
                for i, v in enumerate(rates)
            ],

            hovertemplate="%{customdata}<extra></extra>"
        ),
        row=1,
        col=2
    )

    # ارتفاع نمودار
    style_fig(fig, 340)

    fig.update_layout(
        margin=dict(
            t=55,
            b=80,
            l=10,
            r=10
        )
    )

    # ─────────────────────────────────────────
    # محور Y
    # ─────────────────────────────────────────
    fig.update_yaxes(
        range=[0, 40],
        showticklabels=False
    )

    # ─────────────────────────────────────────
    # تنظیمات عمومی محور X
    # ─────────────────────────────────────────
    fig.update_xaxes(
        tickfont={
            "size": 10,
            "color": MUTED
        }
    )

    # ─────────────────────────────────────────
    # فاصله ارسال:
    # هیچ نوشته‌ای زیر نمودار نمایش داده نشود
    # ─────────────────────────────────────────
    fig.update_xaxes(
        showticklabels=False,
        row=1,
        col=1
    )

    # ─────────────────────────────────────────
    # پاداش خرید:
    # بازه‌های پولی زیر نمودار نمایش داده نشوند
    # ─────────────────────────────────────────
    fig.update_xaxes(
        showticklabels=False,
        row=1,
        col=2
    )

    # ─────────────────────────────────────────
    # استایل عنوان دو نمودار
    # ─────────────────────────────────────────
    fig.update_annotations(
        font={
            "size": 11,
            "color": MUTED,
            "family": "Vazirmatn, Tahoma, sans-serif"
        }
    )

    # ─────────────────────────────────────────
    # متن توضیحی فقط زیر نمودار پاداش خرید
    # ─────────────────────────────────────────
    fig.add_annotation(
        x=1,
        y=-0.18,
        xref="paper",
        yref="paper",
        text=(
            "انتخاب سطح مناسب پاداش خرید می‌تواند به‌عنوان<br>"
            "یکی از اهرم‌های مدیریت ریزش مشتری بررسی شود "
        ),
        showarrow=False,
        align="center",
        font={
            "size": 10,
            "color": MUTED,
            "family": "Vazirmatn, Tahoma, sans-serif"
        }
    )

    return fig


# ─────────────────────────────────────────────
# Shared Plotly styling
# ─────────────────────────────────────────────
def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        font={
            "family": "Vazirmatn, Tahoma, sans-serif",
            "color": INK,
            "size": 12
        },

        margin={
            "l": 6,
            "r": 6,
            "t": 10,
            "b": 6
        },

        showlegend=False,

        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": BORDER,
            "font": {
                "family": "Vazirmatn, Tahoma",
                "color": INK,
                "size": 12
            }
        }
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=BORDER,
        tickfont={
            "color": MUTED,
            "size": 11
        }
    )

    fig.update_yaxes(
        gridcolor="#EEF3FC",
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont={
            "color": MUTED,
            "size": 11
        }
    )

    return fig


# ─────────────────────────────────────────────
# Bar chart helper
# ─────────────────────────────────────────────
def bar_fig(
    labels: list,
    values: list,
    colors: list,
    height: int = 220,
    horizontal: bool = False
) -> go.Figure:

    texts = [f"{fa(v)}٪" for v in values]

    if horizontal:
        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker={
                    "color": colors,
                    "line": {
                        "width": 0
                    }
                },
                width=0.5,
                text=texts,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x}٪<extra></extra>"
            )
        )

        style_fig(fig, height)

        fig.update_xaxes(
            range=[0, max(values) * 1.3],
            showticklabels=False
        )

        fig.update_yaxes(
            gridcolor="rgba(0,0,0,0)"
        )

    else:
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker={
                    "color": colors,
                    "line": {
                        "width": 0
                    }
                },
                width=0.5,
                text=texts,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{y}٪<extra></extra>"
            )
        )

        style_fig(fig, height)

        fig.update_yaxes(
            range=[0, max(values) * 1.28],
            showticklabels=False,
            gridcolor="rgba(0,0,0,0)"
        )

    return fig


# ─────────────────────────────────────────────
# Cashback band labels
# این تابع حذف نشده چون در بخش‌های دیگر برنامه استفاده می‌شود.
# ─────────────────────────────────────────────
def cashback_band_labels(compact_form: bool = True) -> list[str]:
    """Raw bands converted to Toman for display (x2000)."""

    b = [x * 2000 for x in DATASET["cashback_bounds_raw"]]

    edges = [
        fa(f"{v // 1000:,}")
        for v in b
    ]

    starts = [
        fa(f"{(x + 1) * 2000 // 1000:,}")
        for x in DATASET["cashback_bounds_raw"]
    ]

    if compact_form:
        return [
            f"تا {edges[0]} هزار",
            f"{starts[0]}–{edges[1]} هزار",
            f"{starts[1]}–{edges[2]} هزار",
            f"بیش از {edges[2]} هزار"
        ]

    return [
        f"تا {fmt_int(b[0])} تومان",
        f"{fmt_int((DATASET['cashback_bounds_raw'][0] + 1) * 2000)} تا {fmt_int(b[1])} تومان",
        f"{fmt_int((DATASET['cashback_bounds_raw'][1] + 1) * 2000)} تا {fmt_int(b[2])} تومان",
        f"بیش از {fmt_int(b[2])} تومان"
    ]
    
# ------------------------------ HTML fragments ------------------------------ #
def h2(title: str, hint: str = "") -> None:
    small = f"<small>{hint}</small>" if hint else ""
    st.markdown(f'<div class="h2">{title}{small}</div>', unsafe_allow_html=True)


def kpi_html(label: str, value: str, unit: str, sub: str, tone: str = "") -> str:
    unit_html = f"<small>{unit}</small>" if unit else ""
    return (f'<div class="card kpi"><div class="lb">{label}</div><div class="vl {tone}">{value}{unit_html}</div>'
            f'<div class="sb">{sub}</div></div>')


def num_html(items: list[tuple[str, str, str]]) -> str:
    """items: (value, caption, css class r/g/b)."""
    return '<div class="an-nums">' + "".join(f'<div class="an-num {c}"><b>{v}</b>{cap}</div>' for v, cap, c in items) + "</div>"


def analysis_text(kicker: str, title: str, numbers: list, bullets: list[str], takeaway: str) -> str:
    li = "".join(f"<li>{b}</li>" for b in bullets)
    return (f'<div class="an"><div class="an-kicker">{kicker}</div><div class="an-title">{title}</div>{num_html(numbers)}'
            f'<ul>{li}</ul><div class="takeaway"><b>پیام مدیریتی:</b> {takeaway}</div></div>')


def risk_label(risk: float) -> str:
    """Risk text shown only for at-risk customers (never for stable ones)."""
    return f"{fa(int(round(risk)))}٪ ریسک خروج"


STATUS_LABEL = {"red": "بحرانی", "amber": "هشدار", "green": "پایدار"}


def customer_card_html(c: dict) -> str:
    """Minimal radar card: 'id · membership', a status bar and the status badge."""
    tone = tone_of(c["risk"])
    tenure = c.get("tenure")
    tenure_txt = f"{fa(int(tenure))} ماه" if tenure is not None else "نامشخص"
    if tone == "green":  # stable: no exit percentage at all
        badge = '<span class="badge green">🟢 پایدار</span>'
        bar = '<div class="bar green"><span style="width:100%"></span></div>'
    else:
        badge = f'<span class="badge {tone}">{STATUS_LABEL[tone]} · {fa(int(round(c["risk"])))}٪</span>'
        bar = f'<div class="bar {tone}"><span style="width:{max(3, min(100, c["risk"])):.0f}%"></span></div>'
    return (f'<div class="c-name"> شناسه کاربر: {fa(c["id"])}<div style="font-size: 13px; color: #64748B; margin-top: 3px;"> سابقه عضویت: {tenure_txt}</div></div>')


def details_html(c: dict) -> str:
    """Two-column card with all 19 real fields, shown only after the visitor clicks the card button."""
    cells = [f'<div>{label}<b>{esc(value)}</b></div>' for label, value in full_info_rows(c)]
    if len(cells) % 2:
        cells.append('<div></div>')
    tone = tone_of(c["risk"])
    cause = main_cause(c) if tone != "green" else None
    extra = (f'<div class="cause {tone}"  style="margin-top:8px; margin-bottom:10px;"><b>علت اصلی:</b> {esc(cause["title"])} — {esc(cause["detail"])}</div>'
             if cause else "")
    return f'<div class="fg">{"".join(cells)}</div>{extra}'


def value_card_html(value: int) -> str:
    return ('<div class="value-card"><div class="cap">ارزش تخمینی حفظ مشتری</div>'
            f'<div class="big">{fmt_int(value)}<small>تومان</small></div>'
            '<div class="note">برآورد ارزش سالانه حفظ این مشتری بر پایه رفتار خرید</div></div>')


def sms_panel_html(tone: str, coupon: str, discount: int, message: str, registered: bool) -> str:
    """Clean SMS proposal: tier tag, highlighted discount code and the message. No technical fields."""
    offer = LAB_OFFERS[tone]
    body = esc(message).replace(esc(coupon), f'<b style="direction:ltr;display:inline-block">{esc(coupon)}</b>')
    ok = '<div class="sms2-ok">✓ اقدام پیشنهادی فعال شد</div>' if registered else ""
    return (
        f'<div class="sms2"><div class="sms2-head"><b>پیامک پیشنهادی بازگشت مشتری</b><span class="tier {offer["css"]}">{offer["tag"]}</span></div>'
        f'<div class="code-row"><span class="cbox {offer["css"]}">کد تخفیف <b>{esc(coupon)}</b></span>'
        f'<span class="pct">{fa(discount)}٪ تخفیف</span></div>'
        f'<div class="sms2-body">{body}</div>{ok}</div>'
    )


# ===========================================================================
# UI
# ===========================================================================
missing = [p.name for p in (MODEL_PATH, STREAM_PATH, INFERENCE_PATH) if not p.exists()]
if missing:
    st.error("فایل‌های زیر کنار app.py پیدا نشدند: " + "، ".join(missing))
    st.stop()

for _key, _val in {
    "view": VIEW_BI, "registered": set(), "flash": None, "scan_count": 0, "last_scan": 0.0, "store_name": DEFAULT_STORE,
    "auto_running": True, "detail_open": set(), "lab_eval": None, **LAB_DEFAULTS,
}.items():
    st.session_state.setdefault(_key, _val)

try:
    with st.spinner("در حال راه‌اندازی موتور هوشمند تحلیل…"):
        engine = get_engine()
        if len(st.session_state.get("radar", [])) != RADAR_SIZE:
            do_scan()
except Exception as exc:  # friendly message instead of a stack trace at the booth
    st.error(f"راه‌اندازی موتور هوشمند تحلیل ناموفق بود: {exc}")
    st.stop()
    
    
# ============================================================================================================= #
# CSS 
# ============================================================================================================= #    
st.markdown("""
<style>

/* ---------- Brand ---------- */
.brand {
    text-align: center;
    margin-bottom: 30px;
    padding: 34px;
}

.brand h1,
.brand p {
    text-align: center;
}

.brand h1{
    margin-bottom: 12px;
}


/* ---------- Navigation ---------- */
div[data-testid="stRadio"] > div {
    width: 100%;
    display: flex;
    justify-content: flex-end;
    direction: rtl;
    gap: 8px;
}

/* جلوگیری از کش آمدن گزینه‌ها */
div[data-testid="stRadio"] label {
    width: auto !important;
    flex: 0 0 auto !important;
    white-space: nowrap;
}

</style>
""", unsafe_allow_html=True)




st.markdown(compact("""
    <div class="brand"><h1>سامانه تحلیل هوشمند و مدیریت ریزش مشتری</h1>
    <p>پایش رفتار مشتری، پیش‌بینی ریسک خروج و پیشنهاد اقدام بازگشت</p></div>"""), unsafe_allow_html=True)

with st.container(key="seg_nav"):
    st.radio("بخش برنامه", VIEWS, key="view", horizontal=True, label_visibility="collapsed")
view = st.session_state["view"]
registered = st.session_state["registered"]




# =========================================================================== #
# TAB 1: Strategic intelligence (full in-depth showcase)
# =========================================================================== #

if view == VIEW_BI:
    D = DATASET
    kpi_slot = st.container()  # filled after the assumption slider below

    # اسلایدر تعیین تعداد مشتریان سالانه کسب‌وکار
    with st.container(key="panel_assump"):
        total_customers = st.slider(
            "تعداد مشتریان سالانه کسب‌وکار (نفر):",
            min_value=1_000,
            max_value=100_000,
            value=10_000,
            step=1_000,
            key="as_total_cust",
        )   



        st.markdown("""
<style>

/* باکس فرض مالی */
div[data-testid="stVerticalBlock"]:has(div[data-testid="stSlider"]) {
    border-radius: 10px;
    
}

/* متن عنوان Slider */
div[data-testid="stSlider"] label {
    color: #3157B7 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* خود Slider */
div[data-testid="stSlider"] {
    margin-top: -6px !important;
    margin-bottom: -26px !important;
}

</style>
""", unsafe_allow_html=True)
    
    
    

    # ------------------------------------------------------------------ #
    # محاسبه زیان سالانه در خطر بر اساس نرخ ریزش مدل و ارزش میانگین هر مشتری
    # ------------------------------------------------------------------ #
    churn_rate = D["churn_rate"] / 100               # 16.84% نرخ ریزش استخراج‌شده از داده‌ها
    estimated_churners = total_customers * churn_rate # تخمین تعداد افراد ریزش‌کرده
    assumed_value_per_customer = 5_000_000              # ارزش فرضی خرید سالانه هر مشتری (تومان) - قابل تغییر
    loss_num, loss_unit = fmt_toman(estimated_churners * assumed_value_per_customer)
    n_txt, ret_txt = fa(f"{D['n']:,}"), fa(f"{D['retained']:,}")
    cc_txt = fa(f"{D['complain_customers']:,}")
    mean_loyal, mean_churned = cash_toman(D["cashback_mean_loyal_raw"]), cash_toman(D["cashback_mean_churned_raw"])
    band_full = cashback_band_labels(compact_form=False)

    with kpi_slot:
        k = st.columns(4)
        k[0].markdown(kpi_html("نرخ ریزش کل", f"{fa(D['churn_rate'])}٪", "", f"از {n_txt} مشتری پایش‌شده", "red"), unsafe_allow_html=True)
        k[1].markdown(kpi_html("مشتریان خارج‌شده", fa(D["churned"]), "نفر", f"{ret_txt} مشتری ماندگار"), unsafe_allow_html=True)
        k[2].markdown(kpi_html("زیان مالی سالانه در خطر", loss_num, loss_unit, "بر پایهٔ فرض مالی"), unsafe_allow_html=True)
        k[3].markdown(kpi_html("قدرت تشخیص الگوهای ریزش", f"{fa(D['detection_power'])}٪", "", "بر اساس داده‌های پایش‌شده", "teal"), unsafe_allow_html=True)
        
    st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] {
    margin-bottom: 30px;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)        


    st.markdown("""
<style>
.section-title {
    text-align: right !important;
    font-weight: 900 !important;
    margin: 0 0 30px 0 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

    st.markdown(
    '<h4 class="section-title">بینش‌های کلیدی از رفتار مشتریان</h4>',
    unsafe_allow_html=True
)
    
    
    
    sat, comp, cat = D["satisfaction"], D["complain"], D["category"]
    analyses = [
        
        (analysis_text(
            "تحلیل 1 · اهرم شکایت‌ها", "یک شکایت، ریزش را ≈ ۳ برابر می‌کند",
            [(f"{fa(comp['بدون شکایت'])}٪", "ریزش بدون شکایت", "g"), (f"{fa(comp['دارای شکایت'])}٪", "ریزش با شکایت", "r"),
             (f"{fa(D['complain_share_of_churners'])}٪", "سهم شاکیان از خارج‌شدگان", "b")],
            [f"  ( {fa(round(D['complain_customers'] / D['n'] * 100, 1))}٪ از مشتریان) شکایت ثبت کرده‌اند.",
             "بیش از نیمی از کل مشتریان خارج‌شده پیش از رفتن شکایت داشته‌اند؛ شکایت قوی‌ترین هشدار زودهنگام است.",
             "پیشنهاد اجرایی: شکایت‌ها را به‌عنوان هشدار ریزش پایش کنید و پیش از خروج مشتری، مداخله کنید."],
            "پیام مدیریتی: شکایت را به‌عنوان یک سیگنال ریزش جدی بگیرید، نه صرفاً یک مسئله خدماتی."), fig_complaint(), "an2"),
        (analysis_text(
            "تحلیل 2 · ماه‌های نخست؛ نقطه حساس مشتری ", "بحران در ماه نخست همراهی رخ می‌دهد",
            [(f"{fa(D['tenure_mean_loyal'])}", "میانگین سابقه ماندگاران (ماه)", "g"), (f"{fa(D['tenure_mean_churned'])}", "میانگین سابقه خارج‌شدگان (ماه)", "r"),
             (f"{fa(D['tenure']['۰ تا ۱ ماه'])}٪", "ریزش مشتریان ۰ تا ۱ ماهه", "b")],
            [f"{fa(D['first_month_share_of_churners'])}٪ از کل مشتریان خارج‌شده حداکثر ۱ ماه سابقه داشته‌اند.",
             "بعد از ماه دوم، نرخ ریزش به ۵ تا ۱۰٪ می‌رسد و در سابقهٔ بالای ۱۲ ماه حدود ۵٪ است.",
             "ریسک در ابتدای رابطه متمرکز است، نه پراکنده در طول عمر مشتری."],
            "برنامهٔ ۳۰ روز اول (خوش‌آمد، بن خرید دوم، تماس پیگیری) بیشترین بازده را دارد."), fig_tenure(), "an3"),
        (analysis_text(
                    "تحلیل 3 · پاداش خرید و رفتار مشتری", "پاداش خرید بیشتر و ارسال سریع‌تر، وفاداری بیشتر مشتریان",
                    [(f"{fa(D['cashback_rate'][0])}٪", "ریزش با پاداش خرید نامناسب", "r"),
                    (f"{fa(D['cashback_rate'][3])}٪", "ریزش با پاداش خرید مناسب", "g"),
                     (f"{fa(D['distance']['بیش از ۲۰'])}٪", "ریزش در ارسال طولانی تر", "b")],
                    [f"میانگین پاداش خرید مشتریان ماندگار {mean_loyal} در برابر {mean_churned} برای خارج‌شدگان.",
                     "ریزش با افزایش زمان ارسال محصول از ۱۳٫۵٪ به ۲۰٫۴٪ می‌رسد.",
                     f"الگوی «خرید و خداحافظی»: ریزش مشتریان با ۰ تا ۲ روز فاصله از آخرین خرید {fa(D['recency']['۰ تا ۲ روز'])}٪ و بالای ۱۰ روز {fa(D['recency']['بیش از ۱۰ روز'])}٪ است.",
                     "رابطه پاداش خرید و ریزش، یک الگوی آماری است و به‌تنهایی اثرگذاری مستقیم را اثبات نمی‌کند."],
                    "پاداش خرید(سیستم کش بک) پلکانی و ارسال رایگان برای مناطق دور، دو اهرم کم‌هزینهٔ نگهداشت‌اند."), fig_behaviour(), "an5"),
        (analysis_text(
            "تحلیل 4 · ریزش مشتری بر اساس دسته خرید", "نرخ ریزش در دسته‌ کالاهای مختلف یکسان نیست",
            [(f"{fa(cat['Mobile Phone'])}٪", "ریزش در موبایل و دیجیتال", "r"), (f"{fa(cat['Fashion'])}٪", "ریزش در مد و پوشاک", "b"), (f"{fa(cat['Grocery'])}٪", "ریزش در خواروبار", "g")],
            [
             f"خواربار با ریزش {fa(cat['Grocery'])}٪ (وفاداری بالای ۹۵٪) نشان می‌دهد خرید تکرارشونده وفاداری می‌سازد.",
             "تفاوت نرخ ریزش میان دسته‌های خرید نشان می‌دهد رفتار مشتری در همه محصولات یکسان نیست."],
            "برای هر دسته محصولات فروشگاه، متناسب با سطح ریسک آن اقدام کنید."), fig_categories(), "an4"),
        
        (analysis_text(
                    "تحلیل 5 · رضایت مشتری همیشه کافی نیست", "رضایت بالا، اما ریزش همچنان وجود دارد",
                    [(f"{fa(sat[5])}٪", "ریزش با رضایت ۵", "r"), (f"{fa(sat[1])}٪", "ریزش با رضایت 1", "g"), (f"{fa(round(sat[5] / sat[1], 1))}×", "نسبت ریزش", "b")],
                    ["در این داده‌ها، نرخ ریزش در بالاترین سطح رضایت نیز پایین‌تر نشده و از ۱۱٫۵٪ تا ۲۳٫۸٪ تغییر می‌کند.",
                     "«ریزش خاموش»: مشتری بدون هیچ اعتراضی و حتی با امتیاز عالی می‌رود؛ نظرسنجی سنتی این گروه را «وفادار» ثبت می‌کند.",
                     "رضایت باید در کنار سابقه، شکایت و الگوی خرید مشتری بررسی شود؛ یک امتیاز رضایت به‌تنهایی تصویر کاملی از ریسک ریزش نمی‌دهد."],
                    "نمرهٔ رضایت را جایگزین پایش رفتاری نکنید؛ سابقه، شکایت و الگوی خرید را زنده رصد کنید."), fig_satisfaction(), "an1"),
    ]
    for text_html, figure, key in analyses:
        with st.container(key=f"panel_{key}"):
            a, b = st.columns([3, 2])
            a.markdown(text_html, unsafe_allow_html=True)
            b.plotly_chart(figure, config=PLOT_CONFIG, theme=None, key=f"ch_{key}")

    h2("نقشهٔ راه اجرایی ۴ مرحله‌ای")
    # steps = [
    #     ("مرحله ۱ · بازاریابی", "۳۰ روز اول مشتری",
    #      ["ماژول خوش‌آمدگویی + بن خرید دوم", "پیام‌رسانی برای مشتریان با سابقهٔ ≤ ۱ ماه"], "شاخص هدف: کاهش ریزش ماه اول از ۵۱٫۸٪"),
    #     ("مرحله ۲ · پشتیبانی", "بستن سریع شکایت",
    #      ["زمان پاسخ‌دهی ۲ ساعته برای شکایت", "کد دلجویی خودکار پس از ثبت شکایت"], "شاخص هدف: کاهش ریزش شاکیان از ۳۱٫۷٪"),
    #     ("مرحله ۳ · لجستیک", "کاهش اثر فاصله",
    #      ["ارسال رایگان برای مناطق دور از انبار", "بررسی مرکز توزیع محلی"], "شاخص هدف: ریزش فاصلهٔ >۲۰ زیر ۱۷٪"),
    #     ("مرحله ۴ · باشگاه مشتریان", "بازگشت وجه و وفاداری",
    #      ["بازگشت وجه پلکانی برای مشتریان کم‌بازگشت", "تمرکز بر صنف موبایل و دیجیتال"], f"شاخص هدف: کاهش ریزش گروه بازگشت وجه {band_full[0]}"),]
    steps = [
    (
        "مرحله ۱ · شروع رابطه",
        "تمرکز بر مشتریان تازه‌وارد",
        [
            "طراحی مسیر مشوق خرید دوم",
            "پایش ویژه مشتریان با سابقهٔ کمتر از یک ماه"
        ],
        " هدف: کاهش قابل‌توجه ریزش مشتریان در ماه اول٪"
    ),

    (
        "مرحله ۲ · پشتیبانی",
        "تبدیل شکایت به فرصت حفظ مشتری",
        [
            "اولویت‌دهی به مشتریان دارای شکایت",
            "طراحی فرآیند مشخص برای پیگیری و جبران نارضایتی"
        ],
        " هدف: کاهش قابل‌توجه ریزش در میان مشتریان دارای شکایت "
    ),

    (
        "مرحله ۳ · تجربه ارسال",
        "کاهش ریسک ناشی از فاصله ارسال",
        [
            "بررسی راهکارهای بهبود زمان و کیفیت ارسال برای مناطق دور",
            "ارزیابی امکان نزدیک‌تر کردن نقاط توزیع به مشتریان"
        ],
        " هدف: کاهش ریزش مشتریان در مناطق با فاصلهٔ بیشتر از ۲۰کیلومتر"
    ),

    (
        "مرحله ۴ · وفاداری",
        "تقویت پاداش خرید و حفظ مشتری",
        [
            "طراحی پاداش خرید متناسب با ارزش و رفتار مشتری",
            "تمرکز بیشتر بر دسته‌های خرید با نرخ ریزش بالاتر"
        ],
        f"  هدف: کاهش ریزش در گروه‌های دریافت‌کننده پاداش خرید کمتر"
    ),
]
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (st_label, title, bullets, target) in zip(cols, steps):
        col.markdown(f'<div class="card road"><div class="st">{st_label}</div><h4>{title}</h4><ul>'
                     + "".join(f"<li>{b}</li>" for b in bullets) + f'</ul><div class="kp">{target}</div></div>', unsafe_allow_html=True)

# =========================================================================== #
# TAB 2: live radar (top, monitoring only) + retention lab (bottom)
# =========================================================================== #
else:
    st.markdown(EXTRA_CSS, unsafe_allow_html=True)

    # ------------------------- live dual radar ------------------------- #
    def radar_panel() -> None:
        """Fragment: only this panel refreshes (every AUTO_EVERY s). It freezes while a details area is open."""
        opened = st.session_state["detail_open"]
        running = st.session_state["auto_running"] and not opened
        if running and time.time() - st.session_state["last_scan"] >= AUTO_EVERY - 0.5:
            do_scan()
        radar = st.session_state["radar"]

        elapsed = min(float(AUTO_EVERY), max(0.0, time.time() - st.session_state["last_scan"]))
        if running:  # the bar fills in sync with the cycle; alternating names restart the animation on every scan
            fill_style = (f"animation:cdfill{st.session_state['scan_count'] % 2} {AUTO_EVERY}s linear forwards;"
                          f"animation-delay:-{elapsed:.2f}s")
            state_tag = f"به‌روزرسانی خودکار | هر   {fa(AUTO_EVERY)} ثانیه"
        else:
            fill_style = f"width:{elapsed / AUTO_EVERY * 100:.0f}%"
            state_tag = "پایش متوقف است" if not opened else "پایش تا بستن اطلاعات کامل متوقف است"

        with st.container(key="panel_radarhead"):
            c1, c2 = st.columns([3, 1.4])
            c1.markdown(compact(f"""
                <div class="an-title" style="margin-bottom:4px">رادار زنده مشتریان</div>
                <div class="tags"><span class="tag blue"پایش شماره {fa(st.session_state['scan_count'])}</span>
                <span class="tag teal">{state_tag}</span>
                <span class="tag">{fa(RADAR_SIZE)} مشتری در حال پایش</span></div>"""), unsafe_allow_html=True)
            c2.toggle("پایش خودکار", key="auto_running")
            st.markdown(f'<div class="cd"><span class="cd-fill" style="{fill_style}"></span></div>', unsafe_allow_html=True)
        
        st.markdown(
    '<div style="margin-bottom: 20px;"></div>',
    unsafe_allow_html=True
)
        cols = st.columns(RADAR_SIZE)
        for col, customer in zip(cols, radar):
            with col:
                with st.container(key=f"card_{tone_of(customer['risk'])}_{customer['id']}"):
                    st.markdown(customer_card_html(customer), unsafe_allow_html=True)
                    is_open = customer["id"] in opened
                    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
                    st.button("بستن اطلاعات کامل" if is_open else "مشاهده اطلاعات کامل ", key=f"detail_{customer['id']}",
                              on_click=toggle_detail, args=(customer["id"],))
                    if is_open:
                        st.markdown(details_html(customer), unsafe_allow_html=True)
        st.markdown(
    '<div style="margin-bottom: 5px;"></div>',
    unsafe_allow_html=True
)
    st.fragment(run_every=AUTO_EVERY)(radar_panel)()

    # ------------------------- retention lab: static form (no rerun until the button is pressed) ------------------------- #
    st.markdown(
    '<h4 class="section-title">ارزیابی مشتری و پیشنهاد هوشمند بازگشت',
    unsafe_allow_html=True
)
    st.markdown(
    '<div style="margin-bottom: 20px;"></div>',
    unsafe_allow_html=True
)   
    col_out, col_in = st.columns([1.3, 1])  # RTL: the input column is drawn on the right

    with col_in:
        with st.container(key="panel_lab_in"):
            st.markdown('<div class="an-title" style="font-size:16px">ورودی‌های ارزیابی مشتری</div>', unsafe_allow_html=True)
            st.markdown( '<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True )
            with st.form(key="customer_eval_form"):
                a1, a2 = st.columns(2)
                tenure_val = a1.number_input("۱) سابقه همراهی مشتری (ماه)", min_value=0, max_value=60, step=1, key="lab_tenure_n")
                st.markdown( '<div style="margin-bottom: 10px;"></div>', unsafe_allow_html=True )
                days_val = a2.number_input("۲) فاصله از آخرین خرید (روز)", min_value=0, max_value=90, step=1, key="lab_days_n")
                st.markdown( '<div style="margin-bottom: 10px;"></div>', unsafe_allow_html=True )
                b1, b2 = st.columns(2)
                with b1:
                    st.markdown('<div class="lbl">۳) شکایت در ماه اخیر</div>', unsafe_allow_html=True)
                    
                    
                    with st.container(key="seg_complain"):
                        st.markdown("""
                        <style>
                        /* کوچک‌تر کردن گزینه‌های شکایت */
                        div[data-testid="stRadio"] div[role="radiogroup"] {
                            gap: 2px !important;
                        }

                        div[data-testid="stRadio"] div[role="radiogroup"] label {
                            padding: 2px 5px !important;
                            margin: 0 !important;
                            min-height: 0 !important;
                        }

                        /* کوچک‌تر کردن دایره رادیو */
                        div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
                            width: 14px !important;
                            height: 14px !important;
                            transform: scale(0.8) !important;
                        }

                        /* کوچک‌تر کردن متن */
                        div[data-testid="stRadio"] div[role="radiogroup"] label p {
                            font-size: 12px !important;
                            margin: 0 !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        complain_label = st.radio(
                            "شکایت",
                            ["خیر", "بله"],
                            key="lab_complain",
                            horizontal=True,
                            label_visibility="collapsed"
                        )
                
                
                
                with b2:
                    st.markdown('<div class="lbl">۴) امتیاز رضایت مشتری</div>', unsafe_allow_html=True)
                    
                    
                    
                    with st.container(key="seg_sat"):
                        st.markdown("""
                        <style>
                        /* کوچک‌تر کردن گزینه‌های رضایت */
                        div[data-testid="stRadio"] div[role="radiogroup"] {
                            gap: 2px !important;
                        }

                        div[data-testid="stRadio"] div[role="radiogroup"] label {
                            padding: 2px 5px !important;
                            margin: 0 !important;
                            min-height: 0 !important;
                        }

                        /* کوچک‌تر کردن دایره رادیو */
                        div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
                            width: 14px !important;
                            height: 14px !important;
                            transform: scale(0.8) !important;
                        }

                        /* کوچک‌تر کردن عددها */
                        div[data-testid="stRadio"] div[role="radiogroup"] label p {
                            font-size: 12px !important;
                            margin: 0 !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        sat_val = st.radio(
                            "رضایت",
                            [1, 2, 3, 4, 5],
                            format_func=fa,
                            key="lab_sat",
                            horizontal=True,
                            label_visibility="collapsed"
                        )

                        st.markdown(
                            '<div style="margin-bottom: 10px;"></div>',
                            unsafe_allow_html=True
                        )
                        
                        

                        st.markdown(
                            '<div style="margin-bottom: 10px;"></div>',
                            unsafe_allow_html=True
                        )


                c1_, c2_ = st.columns(2)
                cat_label = c1_.selectbox("۵) دسته کالای ترجیحی", list(LAB_CATEGORY_LABELS.keys()), key="lab_cat_l")
                st.markdown( '<div style="margin-bottom: 10px;"></div>', unsafe_allow_html=True )
                orders_val = c2_.number_input("۶) تعداد کل سفارش‌های موفق", min_value=1, max_value=50, step=1, key="lab_orders_n")
                st.markdown( '<div style="margin-bottom: 10px;"></div>', unsafe_allow_html=True )
                tier_label = st.selectbox("۷) سطح شهر مشتری", list(LAB_TIER_LABELS.keys()), key="lab_tier_l")
                st.markdown( '<div style="margin-bottom: 10px;"></div>', unsafe_allow_html=True )
                store_input = st.text_input("۸) نام کسب‌وکار (برای درج در پیامک)", key="store_name")
                st.markdown( '<div style="margin-bottom: 10px;"></div>', unsafe_allow_html=True )
                submitted = st.form_submit_button("ارزیابی هوشمند ریسک ریزش مشتری")
            st.markdown('<div class="mini-title">تست سناریوهای سریع (فقط فرم را پر می‌کنند):</div>', unsafe_allow_html=True)
            st.markdown( '<div style="margin-bottom: 10px;"></div>', unsafe_allow_html=True )
            with st.container(key="presets_zone"):
                p1, p2, p3 = st.columns(3)
                for col, number in ((p1, 1), (p2, 2), (p3, 3)):
                    col.button(LAB_PRESETS[number]["label"], key=f"preset_{number}", on_click=apply_preset, args=(number,))
            st.markdown('<div class="lab-note">سایر فاکتورها با میانهٔ مشتریان مشابه در داده‌ها تکمیل می‌شوند.</div>', unsafe_allow_html=True)
            st.markdown( '<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True )

    if submitted:  # the engine runs only now
        st.session_state["lab_eval"] = {
            "tenure": int(tenure_val), "complain": 1 if complain_label == "بله" else 0, "category": LAB_CATEGORY_LABELS[cat_label],
            "days": int(days_val), "tier": LAB_TIER_LABELS[tier_label], "sat": int(sat_val), "orders": int(orders_val),
            "store": (store_input or DEFAULT_STORE).strip() or DEFAULT_STORE,
        }

    with col_out:
        with st.container(key="panel_lab_out"):
            st.markdown('<div class="an-title" style="font-size:16px">نتیجه ارزیابی و اقدام پیشنهادی</div>', unsafe_allow_html=True)
            ev = st.session_state.get("lab_eval")
            st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)
            if ev is None:
                st.markdown('<div class="helpcard">مشخصات مشتری را در فرم مقابل تکمیل کرده و دکمه محاسبه را بزنید '
                            'تا خروجی و سناریوی پیشنهادی نمایش یابد.</div>', unsafe_allow_html=True)
                st.markdown('<div style="margin-bottom: 25px;"></div>', unsafe_allow_html=True)

            else:
                customer = engine.lab_score(ev["tenure"], ev["complain"], ev["category"], ev["days"], ev["tier"], ev["sat"], ev["orders"])
                tone = tone_of(customer["risk"])
                scenario = customer["scenario"]
                if scenario == "general":
                    scenario = input_scenario(ev["tenure"], ev["complain"], ev["days"])
                lab_id = f"LAB-{zlib.crc32(repr(sorted(ev.items())).encode()) % 90000 + 10000}"
                is_registered = lab_id in registered
                if st.session_state.get("flash") == lab_id:
                    st.session_state["flash"] = None  # show the confirmation exactly once
                    st.toast("سناریوی بازگشت برای مشتری ثبت شد", icon="✅")
                if tone == "green":
                    st.markdown('<div class="stable">🟢 مشتری وفادار و پایدار - بدون ریسک خروج</div>', unsafe_allow_html=True)
                else:
                    offer = LAB_OFFERS[tone]
                    cause = main_cause(customer)
                    cause_txt = f"{esc(cause['title'])} — {esc(cause['detail'])}" if cause else "ترکیب چند عامل رفتاری"
                    st.markdown(compact(f"""
                        <div class="riskbox {tone}"><div class="riskline {tone}">{risk_label(customer['risk'])}
                        <span class="badge {tone}" style="margin-inline-start:auto">{STATUS_LABEL[tone]}</span></div>
                        <div class="bar {tone}"><span style="width:{max(3, min(100, customer['risk'])):.0f}%"></span></div>
                        <div class="cause {tone}"><b>علت اصلی ریزش:</b> {cause_txt}</div>
                        <div class="cause {tone}"><b>پیشنهاد مداخله:</b> {esc(SCENARIOS[scenario]['label'])}</div></div>"""),
                        unsafe_allow_html=True)
                    st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)

                    st.markdown(value_card_html(retention_value(ev["category"], customer["risk"])), unsafe_allow_html=True)
                    st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)

                    message = build_offer_sms(scenario, ev["store"], offer["code"], offer["discount"])
                    st.markdown(sms_panel_html(tone, offer["code"], offer["discount"], message, is_registered), unsafe_allow_html=True)
                    st.markdown('<div style="margin-bottom: 25px;"></div>', unsafe_allow_html=True)


                
                    with st.container(key="register_zone"):
                        if is_registered:
                            st.button("✅ اقدام ثبت شد", disabled=True, key=f"reg_{lab_id}")
                        else:
                            st.button("ثبت اقدام پیشنهادی", key=f"reg_{lab_id}", on_click=register_scenario, args=(lab_id,))

                
                st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)


#st.markdown('<div class="foot">برای بازدیدکنندهٔ بعدی، نمایش را از ابتدا شروع کنید.</div>', unsafe_allow_html=True)
_, mid, _ = st.columns([2, 1, 2])
mid.button("بازنشانی نمایش", on_click=reset_demo, key="reset_btn")
