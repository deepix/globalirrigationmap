# Creates a sample of (features, labels) as a multi-band image on Google Earth Engine
# Requires: sample points for which to fill in features as a feature collection
# Requires: labels asset as an image

import ee

from common import (model_scale, wait_for_task_completion, get_features_image, get_labels, model_snapshot_path_prefix,
                    export_asset_table_to_drive, model_projection, num_samples)
from sampler import get_worldwide_sample_points


def create_all_features_labels_image(region_fc, model_year):
    images = get_features_image(region_fc, model_year, "all")
    labels_image = get_labels(region_fc)
    images.append(labels_image)
    lonlat_image = ee.Image.pixelLonLat() \
        .clipToCollection(region_fc) \
        .select(['longitude', 'latitude'], ['X', 'Y'])
    images.append(lonlat_image)
    features_labels_image = ee.Image.cat(*images)
    return features_labels_image


def main():
    model_year = '2000'     # Siebert labels

    # Step 1/3: create or fetch sample points
    asset_description = f'training_sample{num_samples}_all_features_labels'
    image_asset_id = f'{model_snapshot_path_prefix}_{asset_description}_image'
    table_asset_id = f'{model_snapshot_path_prefix}_{asset_description}_table'
    ee.Initialize()
    sample_points_fc = get_worldwide_sample_points()

    # Step 2/3: sample all features into an image
    features_labels_image = create_all_features_labels_image(sample_points_fc, model_year)
    task = ee.batch.Export.image.toAsset(
        crs=model_projection,
        image=features_labels_image,
        scale=model_scale,
        assetId=image_asset_id,
        description=asset_description
    )
    task.start()
    wait_for_task_completion([task], exit_if_failures=True)

    # Step 3/3: convert image into a table
    features_labels_image = ee.Image(image_asset_id)
    # For training sample, it is more efficient to export a table than a raster with (mostly) 0's
    training_fc = features_labels_image.sampleRegions(
        collection=sample_points_fc,
        projection=model_projection,
        scale=model_scale,
        geometries=True
    )
    task = ee.batch.Export.table.toAsset(
        collection=training_fc,
        assetId=table_asset_id,
        description=asset_description.replace('/', '_')
    )
    task.start()
    wait_for_task_completion([task], exit_if_failures=True)

    # Step 3a: export to drive for offline model development
    export_asset_table_to_drive(table_asset_id)


if __name__ == '__main__':
    main()
