# import ee
# import geemap
# import pandas as pd
# import json
# import matplotlib.pyplot as plt

# ee.Authenticate()
# ee.Initialize(project="arctic-bee-351203")

# dataset = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07")
# print(dataset.size().getInfo())

# first_image = dataset.first()

# # Get information about the bands (columns)
# band_names = first_image.bandNames().getInfo()

# print("Available bands (columns) in the dataset:")
# print(band_names)


# start_date = dataset.first().get('system:time_start')
# end_date = dataset.sort('system:time_start', False).first().get('system:time_start')
# print('Start Date:', start_date.getInfo())
# print('End Date:', end_date.getInfo())

# import datetime

# # Start and end timestamps in milliseconds
# start_timestamp = start_date.getInfo()  # Start date
# end_timestamp = end_date.getInfo()   # End date

# # Convert milliseconds to seconds for datetime
# start_date = datetime.datetime.fromtimestamp(start_timestamp / 1000.0)
# end_date = datetime.datetime.fromtimestamp(end_timestamp / 1000.0)

# print('Start Date:', start_date)
# print('End Date:', end_date)


# import ee
# import csv
# from datetime import datetime
# import ast
# import pandas as pd

# # Initialize Earth Engine
# ee.Initialize()

# # Function to read district coordinates from CSV
# def read_district_coordinates(file_path):
#     districts = {}
#     with open(file_path, 'r') as file:
#         csv_reader = csv.DictReader(file)
#         for row in csv_reader:
#             district = row['District']
#             coords = ast.literal_eval(row['Coordinates'])
#             districts[district] = coords
#     return districts

# # Load district coordinates
# district_coords = read_district_coordinates('bangladesh_district_coordinates.csv')
# print(f"Number of districts loaded: {len(district_coords)}")

# # Load the GPM monthly ImageCollection
# dataset = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07")
# print(f"Number of images in dataset: {dataset.size().getInfo()}")

# # Function to calculate metrics for a district
# def calculate_district_metrics(img, district_name, geometry):
#     roi = ee.Geometry.Polygon(geometry)
#     metrics = img.reduceRegion(
#         reducer=ee.Reducer.mean(),
#         geometry=roi,
#         scale=10000,
#         bestEffort=True
#     )
#     return ee.Feature(None, metrics).set({
#         'system:time_start': img.get('system:time_start'),
#         'district': district_name
#     })

# # Process data for all districts
# all_district_data = []

# for district, coords in district_coords.items():
#     print(f"Processing district: {district}")
#     roi = ee.Geometry.Polygon(coords)

#     # Apply the function to the ImageCollection for this district
#     district_data = dataset.map(lambda img: calculate_district_metrics(img, district, coords))

#     # Convert to FeatureCollection and get info
#     district_fc = ee.FeatureCollection(district_data)
#     district_info = district_fc.getInfo()

#     print(f"Number of features for district {district}: {len(district_info['features'])}")

#     for feature in district_info['features']:
#         properties = feature['properties']
#         timestamp = datetime.fromtimestamp(properties['system:time_start'] / 1000)

#         # Extract all the metrics
#         precipitation = properties.get('precipitation')
#         gauge_relative_weighting = properties.get('gaugeRelativeWeighting')
#         precipitation_quality_index = properties.get('precipitationQualityIndex')
#         random_error = properties.get('randomError')

#         if precipitation is not None:
#             data = {
#                 'district': properties['district'],
#                 'system:time_start': timestamp,
#                 'precipitation': precipitation,
#                 'year': timestamp.year,
#                 'month': timestamp.month,
#                 'day': timestamp.day,
#                 'gaugeRelativeWeighting': gauge_relative_weighting,
#                 'precipitationQualityIndex': precipitation_quality_index,
#                 'randomError': random_error,
#                 'SW_Lon': coords[0][0],
#                 'SW_Lat': coords[0][1],
#                 'NE_Lon': coords[2][0],
#                 'NE_Lat': coords[2][1],
#                 'NW_Lon': coords[0][0],
#                 'NW_Lat': coords[2][1],
#                 'SE_Lon': coords[2][0],
#                 'SE_Lat': coords[0][1]
#             }
#             all_district_data.append(data)

# print(f"Total records processed: {len(all_district_data)}")

# # get all_district_data file size and then download in csv


# # Convert to DataFrame
# df = pd.DataFrame(all_district_data)
# # download df in csv
# df.to_csv('precipitation_data.csv', index=False)
# # save the csv in drive



# df = pd.read_csv("/content/GPM_precipitation_BD_export.csv")

# # View the first few rows
# print(df.head())


# def extract_coordinates(geo_json):
#     # Parse the JSON string
#     geo_data = json.loads(geo_json)

#     # Extract coordinates
#     coords = geo_data['coordinates'][0]

#     # Ensure we have exactly 5 coordinate pairs (including the closing pair)
#     if len(coords) != 5:
#         raise ValueError(f"Expected 5 coordinate pairs, but found {len(coords)}")

#     # Extract the corners (ignoring the closing pair which is the same as the first)
#     sw = coords[0]
#     nw = coords[1]
#     ne = coords[2]
#     se = coords[3]

#     return sw[0], sw[1], nw[0], nw[1], ne[0], ne[1], se[0], se[1]

# # Apply the function to create new columns
# df[['SW_Lon', 'SW_Lat', 'NW_Lon', 'NW_Lat', 'NE_Lon', 'NE_Lat', 'SE_Lon', 'SE_Lat']] = df['.geo'].apply(lambda x: pd.Series(extract_coordinates(x)))

# # Drop the original '.geo' and 'system:index' columns
# df.drop(columns=['system:index', '.geo'], inplace=True)

# # Display the first few rows of the updated dataframe
# print(df.head())

# # If you want to save the updated dataframe to a new CSV file, uncomment the following line:
# df.to_csv('updated_dataset.csv', index=False)