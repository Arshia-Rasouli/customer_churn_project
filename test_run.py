import pandas as pd
from churn_inference import predict_single_customer

# خواندن فایل داده‌های شبیه‌ساز دمو
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STREAM_PATH = BASE_DIR / "data" / "test_stream_data.csv"

df = pd.read_csv(STREAM_PATH)
# انتخاب یک مشتری نمونه از داده‌ها
sample_customer = df.iloc[0].to_dict()

# فراخوانی موتور پیش‌بینی هوشمند
result = predict_single_customer(sample_customer)

print("--- نتیجه آنالیز مشتری ---")
print(f"شناسه کاربر: {sample_customer.get('CustomerID')}")
print(f"احتمال ریزش: {result['churn_probability_pct']:.2f}%")
print("محرک‌های اصلی ریسک (Top Risk Drivers):")
for driver in result["top_risk_drivers"]:
    print(f" - عامل: {driver['feature']} | شدت اثر: {driver['impact_pct_points']:.2f}%") 