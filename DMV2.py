# --- Import libraries ---
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!pip install folium

import folium

# --- API setup ---
API_KEY = "e0b7cd00b3a896c7494ae2889f114bd5"
city = input("Enter city name: ")

# --- Fetch weather data ---
url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
data = requests.get(url).json()

if "list" not in data:
    print(" City not found or API limit exceeded.")
    print(data)
    exit()
else:
    print(f"Weather data fetched for {city}")

# --- Create DataFrame ---
df = pd.DataFrame([{
    "Datetime": item["dt_txt"],
    "Temperature": item["main"]["temp"],
    "Humidity": item["main"]["humidity"],
    "WindSpeed": item["wind"]["speed"],
    "Weather": item["weather"][0]["description"]
} for item in data["list"]])

df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Date"] = df["Datetime"].dt.date

print("\nSample Data:")
print(df.head())

# --- Daily Summary ---
daily = df.groupby("Date").agg({
    "Temperature": ["mean", "max", "min"],
    "Humidity": "mean",
    "WindSpeed": "mean"
}).round(1).reset_index()

daily.columns = ["Date", "AvgTemp", "MaxTemp", "MinTemp", "AvgHumidity", "AvgWind"]
print("\nDaily Weather Summary:")
print(daily)

# --- City coordinates ---
lat, lon = data['city']['coord']['lat'], data['city']['coord']['lon']
print(f"\n City Coordinates: Lat: {lat}, Lon: {lon}")

# --- Create Weather Map ---
weather_map = folium.Map(location=[lat, lon], zoom_start=10)
current = df.iloc[0]
popup = f"<b>{city}</b><br>Temp: {current['Temperature']}°C<br>Humidity: {current['Humidity']}%"
folium.Marker([lat, lon], popup=popup, tooltip=f"{city} Weather").add_to(weather_map)
weather_map.save("weather_map.html")
print(" Weather map created → weather_map.html")

# --- Plots ---

# Temperature trend
plt.figure(figsize=(10,4))
plt.plot(df["Datetime"], df["Temperature"], marker='o')
plt.title(f"Temperature Trend - {city}")
plt.xlabel("Time")
plt.ylabel("Temp (°C)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Daily average temperature
plt.figure(figsize=(8,4))
plt.bar(daily["Date"].astype(str), daily["AvgTemp"], color='skyblue')
plt.title(f"Daily Avg Temperature - {city}")
plt.xlabel("Date")
plt.ylabel("Avg Temp (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Temperature vs Humidity
plt.figure(figsize=(5,4))
plt.scatter(df["Temperature"], df["Humidity"], color='orange')
plt.title("Temp vs Humidity")
plt.xlabel("Temp (°C)")
plt.ylabel("Humidity (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(df[["Temperature", "Humidity", "WindSpeed"]].corr(),
            annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
