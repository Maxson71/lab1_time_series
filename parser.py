import requests
import pandas as pd
from datetime import date, timedelta
import os

end = date.today()
start = end - timedelta(days=3*365)
valcode = "CNY"

# Завантаження з офіційного API НБУ (JSON на діапазон дат)
url = "https://bank.gov.ua/NBU_Exchange/exchange_site"
params = {
    "start": start.strftime("%Y%m%d"),
    "end": end.strftime("%Y%m%d"),
    "valcode": valcode.lower(),
    "sort": "exchangedate",
    "order": "asc",
    "json": "",
}
resp = requests.get(url, params=params)
resp.raise_for_status()
data = resp.json()

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["exchangedate"], format="%d.%m.%Y")
df = df.sort_values("date").reset_index(drop=True)
df = df[["date", "cc", "rate"]]

os.makedirs(valcode + "/data", exist_ok=True)
# Збереження у файл (CSV + Parquet)
df.to_csv(f"{valcode}/data/NBU_{valcode}_UAH.csv", index=False)
try:
    df.to_parquet(f"NBU_{valcode}_UAH.csv", index=False)
except Exception:
    pass
print(df.head())