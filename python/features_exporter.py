# Creates a sample of (features, labels) as a multi-band image on Google Earth Engine
# Requires: sample points for which to fill in features as a feature collection
# Requires: labels asset as an image

import ee

from common import (model_scale, wait_for_task_completion, get_selected_features_image, model_snapshot_path_prefix,
                    model_projection)


def export_selected_features_for_year(model_year):
    asset_description = f'features_{model_year}'
    asset_name = f'{model_snapshot_path_prefix}_{asset_description}'
    features_image = get_selected_features_image(model_year)
    task = ee.batch.Export.image.toAsset(
        image=features_image,
        description=asset_description,
        assetId=asset_name,
        crs=model_projection,
        # default is 1000: don't want this!
        scale=model_scale
    )
    task.start()
    return task


def main():
    ee.Initialize()
    model_years = range(2001, 2016)
    tasks = []
    for year in model_years:
        task = export_selected_features_for_year(str(year))
        tasks.append(task)
    wait_for_task_completion(tasks)


if __name__ == '__main__':
    main()
