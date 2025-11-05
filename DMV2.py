
!pip install folium
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Enter your API key & city
api_key = "e0b7cd00b3a896c7494ae2889f114bd5"
city = input("Enter city name: ")

url = "http://api.openweathermap.org/data/2.5/forecast"

params = {
    "q": city,
    "appid": api_key,
    "units": "metric"
}

response = requests.get(url, params=params)
data = response.json()

if "list" not in data:
    print("City not found or API limit exceeded")
    print(data)
else:
    print(f"Weather data fetched for {city}")


weather_list = data["list"]

records = []
for entry in weather_list:
    records.append({
        "Datetime": entry["dt_txt"],
        "Temperature": entry["main"]["temp"],
        "Humidity": entry["main"]["humidity"],
        "WindSpeed": entry["wind"]["speed"],
        "Weather": entry["weather"][0]["description"]
    })


df = pd.DataFrame(records)
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Date"] = df["Datetime"].dt.date


df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("\n Sample Data:")
print(df.head())


daily = df.groupby("Date").agg({
    "Temperature": ["mean", "max", "min"],
    "Humidity": "mean",
    "WindSpeed": "mean"
}).reset_index()

daily.columns = ["Date", "AvgTemp", "MaxTemp", "MinTemp", "AvgHumidity", "AvgWind"]

print("\nDaily Weather Summary:")
print(daily)


plt.figure(figsize=(10,4))
plt.plot(df["Datetime"], df["Temperature"], marker='o')
plt.title(f"Temperature Trend - {city}")
plt.xlabel("Time")
plt.ylabel("Temp (°C)")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.bar(daily["Date"].astype(str), daily["AvgTemp"])
plt.title(f"Daily Avg Temperature - {city}")
plt.xlabel("Date")
plt.ylabel("Avg Temp (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.scatter(df["Temperature"], df["Humidity"])
plt.title("Temp vs Humidity")
plt.xlabel("Temp (°C)")
plt.ylabel("Humidity (%)")
plt.grid()
plt.show()

plt.figure(figsize=(5,4))
sns.heatmap(df[["Temperature", "Humidity", "WindSpeed"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
