from shapely.geometry import Point
from pyproj import Transformer
from typing import TypedDict


# Define the return type using TypedDict
class BoundingBox(TypedDict):
    sw_lon: float
    sw_lat: float
    ne_lon: float
    ne_lat: float
    nw_lon: float
    nw_lat: float
    se_lon: float
    se_lat: float


# Function to buffer a point and get bounding box coordinates
def get_buffered_bounding_box(lat: float, lon: float, buffer_km: float) -> BoundingBox:
    # Step 1: Convert lat/lon to a UTM point
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    x, y = transformer_to_utm.transform(lon, lat)

    # Step 2: Create a point and buffer it by buffer_km (converted to meters)
    point = Point(x, y)
    buffer_meters = buffer_km * 1000  # convert km to meters
    buffered_point = point.buffer(buffer_meters)

    # Step 3: Get the bounding box (minx, miny, maxx, maxy)
    bounds = buffered_point.bounds
    minx, miny, maxx, maxy = bounds

    # Step 4: Convert the bounding box corners back to lat/lon
    transformer_to_latlon = Transformer.from_crs(
        "EPSG:32633", "EPSG:4326", always_xy=True
    )
    sw_lon, sw_lat = map(float, transformer_to_latlon.transform(minx, miny))
    ne_lon, ne_lat = map(float, transformer_to_latlon.transform(maxx, maxy))
    nw_lon, nw_lat = map(float, transformer_to_latlon.transform(minx, maxy))
    se_lon, se_lat = map(float, transformer_to_latlon.transform(maxx, miny))

    return BoundingBox(
        sw_lon=sw_lon,
        sw_lat=sw_lat,
        ne_lon=ne_lon,
        ne_lat=ne_lat,
        nw_lon=nw_lon,
        nw_lat=nw_lat,
        se_lon=se_lon,
        se_lat=se_lat,
    )


# Example usage
lat = 20.7128  # Latitude of the point (e.g., New York City)
lon = -74.0060  # Longitude of the point
buffer_km = 1  # Buffer in kilometers

bbox: BoundingBox = get_buffered_bounding_box(lat, lon, buffer_km)
print(bbox)
