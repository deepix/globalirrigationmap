import ee
from common import base_asset_directory, region_boundaries, export_image_to_asset, wait_for_task_completion, \
    model_snapshot_version


def combine_maps(cropland_image, non_cropland_image):
    non_cl_mask = ee.Image(f"{base_asset_directory}/nonCLMask")
    non_cropland_image = non_cropland_image.mask(non_cl_mask)
    # Fix a labelling mistake: uses class 3 instead of 2
    cropland_image = cropland_image.expression("classification = b(0) == 3 ? 2 : b(0)")
    combined_image = cropland_image.mask(cropland_image) \
        .blend(non_cropland_image.mask(non_cropland_image)) \
        .unmask(0) \
        .clipToCollection(region_boundaries("world"))
    return combined_image


years = range(2001, 2016)


def main():
    tasks = []
    for year in years:
        cropland_image = ee.Image(f"users/deepakna/ellecp/{year}_ternary")
        non_cropland_image = ee.Image(f"{base_asset_directory}/{model_snapshot_version}_results_{year}")
        combined_image = combine_maps(cropland_image, non_cropland_image)
        task = export_image_to_asset(combined_image, f"{model_snapshot_version}_combined_{year}")
        tasks.append(task)
    wait_for_task_completion(tasks)


if __name__ == '__main__':
    ee.Initialize()
    main()
