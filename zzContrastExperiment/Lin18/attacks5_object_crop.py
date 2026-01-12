from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import numpy as np
import os


def attacks5_object_crop(shpFile, outshpfile, axis):
    """
    attacks5_object_crop - Crop a vector map (shapefile) by half along X or Y axis.

    Parameters:
    shpFile (str): Path to the input shapefile.
    outshpfile (str): Name for the output shapefile.
    axis (str): Axis to crop along ('X' or 'Y').

    Returns:
    str: Full path of the saved output shapefile.
    """
    # Read the shapefile using geopandas
    try:
        gdf = gpd.read_file(shpFile)
    except Exception as e:
        raise ValueError(f"无法读取文件，错误: {e}")

    # Collect all X and Y coordinates from the geometries
    all_x = []
    all_y = []

    for geom in gdf.geometry:
        if geom is not None and geom.is_valid:
            if geom.geom_type == 'Point':
                # For Point geometries, use xy attribute
                x, y = geom.xy
                all_x.extend(x)
                all_y.extend(y)
            elif geom.geom_type == 'LineString':
                # For LineString geometries, use xy attribute
                x, y = geom.xy
                all_x.extend(x)
                all_y.extend(y)
            elif geom.geom_type == 'Polygon':
                # For Polygon geometries, use exterior.xy
                x, y = geom.exterior.xy
                all_x.extend(x)
                all_y.extend(y)
                # If the polygon has interior rings, include those too
                for interior in geom.interiors:
                    x, y = interior.xy
                    all_x.extend(x)
                    all_y.extend(y)
            elif geom.geom_type == 'MultiPolygon':
                # For MultiPolygon geometries, iterate over each individual polygon
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    all_x.extend(x)
                    all_y.extend(y)
                    for interior in poly.interiors:
                        x, y = interior.xy
                        all_x.extend(x)
                        all_y.extend(y)

    # Calculate the median value for the selected axis
    if axis.lower() == 'x':
        mid_value = np.median(all_x)
    elif axis.lower() == 'y':
        mid_value = np.median(all_y)
    else:
        raise ValueError('axis 参数无效，请使用 "X" 或 "Y"。')

    # Initialize a list to hold the cropped geometries
    cropped_geometries = []

    for geom in gdf.geometry:
        if geom is not None and geom.is_valid:
            if geom.geom_type == 'Point':
                x, y = geom.xy
                if axis.lower() == 'x':
                    if x[0] <= mid_value:
                        cropped_geometries.append(geom)
                    else:
                        cropped_geometries.append(None)  # Placeholder for invalid geometry
                else:
                    if y[0] <= mid_value:
                        cropped_geometries.append(geom)
                    else:
                        cropped_geometries.append(None)  # Placeholder for invalid geometry
            elif geom.geom_type == 'LineString':
                x, y = geom.xy
                if axis.lower() == 'x':
                    x_crop = [xi for xi in x if xi <= mid_value]
                    y_crop = y[:len(x_crop)]
                else:
                    y_crop = [yi for yi in y if yi <= mid_value]
                    x_crop = x[:len(y_crop)]

                # Only create LineString if there are at least 2 points
                if len(x_crop) > 1 and len(y_crop) > 1:
                    cropped_geometries.append(LineString(zip(x_crop, y_crop)))
                else:
                    cropped_geometries.append(None)  # Placeholder for invalid geometry
            elif geom.geom_type == 'Polygon':
                x, y = geom.exterior.xy
                if axis.lower() == 'x':
                    x_crop = [xi for xi in x if xi <= mid_value]
                    y_crop = y[:len(x_crop)]
                else:
                    y_crop = [yi for yi in y if yi <= mid_value]
                    x_crop = x[:len(y_crop)]

                # Ensure there are enough points to form a valid polygon
                if len(x_crop) > 2 and len(y_crop) > 2:
                    if x_crop[0] != x_crop[-1]:
                        x_crop.append(x_crop[0])
                        y_crop.append(y_crop[0])
                    cropped_geometries.append(Polygon(zip(x_crop, y_crop)))
                else:
                    cropped_geometries.append(None)  # Placeholder for invalid geometry
            elif geom.geom_type == 'MultiPolygon':
                poly_geometries = []
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    if axis.lower() == 'x':
                        x_crop = [xi for xi in x if xi <= mid_value]
                        y_crop = y[:len(x_crop)]
                    else:
                        y_crop = [yi for yi in y if yi <= mid_value]
                        x_crop = x[:len(y_crop)]

                    # Only create Polygon if there are at least 3 points
                    if len(x_crop) > 2 and len(y_crop) > 2:
                        if x_crop[0] != x_crop[-1]:
                            x_crop.append(x_crop[0])
                            y_crop.append(y_crop[0])
                        poly_geometries.append(Polygon(zip(x_crop, y_crop)))
                if poly_geometries:
                    cropped_geometries.append(poly_geometries)
                else:
                    cropped_geometries.append(None)  # Placeholder for invalid geometry

    # Ensure that cropped_geometries has the same length as the original DataFrame
    while len(cropped_geometries) < len(gdf):
        cropped_geometries.append(None)  # Fill missing values with None

    # Create a new GeoDataFrame with the cropped geometries
    cropped_gdf = gdf.copy()
    cropped_gdf.geometry = cropped_geometries

    # Prepare the output path
    output_dir = os.path.join('attacked', 'cropped')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'half_cropped_{outshpfile}')

    # Save the cropped GeoDataFrame to the new shapefile
    cropped_gdf.to_file(output_path)

    print(f'一半裁剪攻击完成，文件已保存到 {output_path}')
    return output_path
