from dotenv import load_dotenv
import os
import json
import urllib.request
import urllib.parse
import ssl
load_dotenv()
API_KEY = os.getenv('AccuWeatherAPIKey')
# 判斷城市名稱的語言
def detect_language(city_name):
    if any(u'\u4e00' <= char <= u'\u9fff' for char in city_name):  # 檢查是否包含中文字符
        return "zh-tw"
    return "en-us"
# 查詢城市的 Location Key
def get_location_key(city_name):
    language = detect_language(city_name)
    base_url = "https://dataservice.accuweather.com/locations/v1/cities/search"
    params = {
        "apikey": API_KEY,
        "q": city_name,
        "language": language
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    # 建立 SSL context 使用系統預設 CA 憑證
    context = ssl.create_default_context()

    with urllib.request.urlopen(url, context=context) as response:
        data = json.loads(response.read().decode())
        if data:
            return data[0]["Key"], data[0]["LocalizedName"]
        else:
            raise ValueError(f"找不到地點：{city_name}")
# 根據 Location Key 查詢當前天氣
def get_current_weather(location_key, language="en-us"):
    base_url = f"https://dataservice.accuweather.com/currentconditions/v1/{location_key}"
    params = {
        "apikey": API_KEY,
        "language": language,
        "details": "true"
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    # 建立 SSL context 使用系統預設 CA 憑證
    context = ssl.create_default_context()

    with urllib.request.urlopen(url, context=context) as response:
        return json.loads(response.read().decode())
# 查詢指定城市的天氣資訊
def get_weather_by_city(city_name):
    try:
        language = detect_language(city_name)
        location_key, city_name = get_location_key(city_name)
        weather = get_current_weather(location_key, language)[0]

        return f"The current weather in {city_name} is {weather['WeatherText']} with a temperature of {weather['Temperature']['Metric']['Value']}°C, humidity {weather['RelativeHumidity']}%, and wind speed {weather['Wind']['Speed']['Metric']['Value']} km/h."
    except Exception as e:
        return f"Unable to retrieve weather information: {str(e)}"