import geopandas as gpd
import numpy as np
import os

def attacks7_geometric(shp_file, outshp_file, angle, scale_factor, x_shift, y_shift):
    """
    attacks7_geometric - Perform combined geometric attacks (rotation, scaling, translation) on vector data.

    Parameters:
    shp_file (str): Path to the input .shp file
    outshp_file (str): Output shapefile name
    angle (float): Rotation angle (in degrees)
    scale_factor (float): Scaling factor (>1 for enlargement, <1 for reduction)
    x_shift (float): Shift along the X axis
    y_shift (float): Shift along the Y axis

    Returns:
    str: Path to the saved shapefile with the geometric transformation applied
    """

    try:
        # Read the shapefile
        shp_data = gpd.read_file(shp_file)
    except Exception as e:
        raise ValueError("Unable to read file, please check the file path.") from e

    # Convert the angle to radians
    theta = np.radians(angle)

    # Perform transformations on each geometry
    transformed_geometries = []

    for geometry in shp_data.geometry:
        # Extract the x and y coordinates
        if geometry is None or geometry.is_empty:
            transformed_geometries.append(geometry)
            continue

        # Ensure the geometry is in a format that can be manipulated (e.g., Point, Polygon, etc.)
        if geometry.geom_type == 'Point':
            x_original, y_original = geometry.x, geometry.y
        elif geometry.geom_type == 'Polygon':
            # For polygons, extract the exterior coordinates (boundary)
            x_original, y_original = geometry.exterior.xy
        elif geometry.geom_type == 'LineString':
            # For linestrings, extract coordinates
            x_original, y_original = geometry.xy

        # Convert to numpy arrays for easier math
        x_original = np.array(x_original)
        y_original = np.array(y_original)

        # Rotate the coordinates
        x_rot = x_original * np.cos(theta) - y_original * np.sin(theta)
        y_rot = x_original * np.sin(theta) + y_original * np.cos(theta)

        # Scale the coordinates
        x_scaled = x_rot * scale_factor
        y_scaled = y_rot * scale_factor

        # Translate the coordinates
        x_trans = x_scaled + x_shift
        y_trans = y_scaled + y_shift

        # Create the new geometry
        if geometry.geom_type == 'Point':
            new_geometry = geometry.__class__((x_trans[0], y_trans[0]))
        elif geometry.geom_type == 'Polygon':
            new_geometry = geometry.__class__(geometry.exterior.__class__(list(zip(x_trans, y_trans))))
        elif geometry.geom_type == 'LineString':
            new_geometry = geometry.__class__(list(zip(x_trans, y_trans)))

        transformed_geometries.append(new_geometry)

    # Add the transformed geometries to a new GeoDataFrame
    transformed_shp = gpd.GeoDataFrame(shp_data, geometry=transformed_geometries)

    # Create the output directory if it doesn't exist
    output_dir = os.path.join('attacked', 'geometric_combined')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the full output path
    output_path = os.path.join(output_dir, f"{angle}_{scale_factor}_{x_shift}_{y_shift}_{outshp_file}")

    # Save the transformed shapefile
    transformed_shp.to_file(output_path)

    print(f"Geometric transformation completed. File saved to {output_path}")

    return output_path
