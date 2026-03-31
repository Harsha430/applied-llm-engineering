import requests
from langchain_core.tools import tool

OPEN_WEATHERAMP_API="52a88e2fc588603f9a44dce6f330a307"

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location using OpenStreetMap for geocoding and OpenWeatherMap for weather."""
    try:
        # Step 1: Use OpenStreetMap (Nominatim) to get coordinates (no API key required)
        geocode_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
        headers = {'User-Agent': 'ToolBasedAgent/1.0'}
        geo_response = requests.get(geocode_url, headers=headers).json()
        
        if not geo_response:
            return f"Error: Could not find coordinates for location {location}"
            
        lat = geo_response[0]['lat']
        lon = geo_response[0]['lon']
        
        # Step 2: Use OpenWeatherMap API key (in file only) to get weather
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPEN_WEATHERAMP_API}&units=metric"
        weather_response = requests.get(weather_url).json()
        
        if weather_response.get("cod") != 200:
            return f"Error: {weather_response.get('message', 'Failed to fetch weather data.')}"
            
        weather_desc = weather_response['weather'][0]['description']
        temp = weather_response['main']['temp']
        
        return f"The current weather in {location} is {weather_desc} with a temperature of {temp}°C."
    except Exception as e:
        return f"An error occurred while fetching the weather: {str(e)}"
