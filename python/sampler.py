import ee
from common import (region_boundaries, model_scale, wait_for_task_completion, model_projection, base_asset_directory,
                    export_asset_table_to_drive, num_samples, train_seed)


world_regions = [
    "North America",
    "Central America",
    "Caribbean",
    "South America",
    "Europe",
    "Africa",
    "SW Asia",
    "Central Asia",
    "N Asia",
    "E Asia",
    "SE Asia",
    "S Asia",
    "Australia",
    # Oceania causes geometry explosion
    # "Oceania"
    # get 2 countries from there instead
    "NZ",
    "PP"
]


def get_total_area():
    all_regions = ee.FeatureCollection(list(map(region_boundaries, world_regions))).flatten()
    # compute this before doing anything else
    total_area = all_regions.aggregate_sum('areaHa').getInfo()
    return total_area


def read_sample(asset_name):
    try:
        sample_fc = ee.FeatureCollection(asset_name)
        # force materialization of feature collection
        _ = sample_fc.limit(10).getInfo()
        # it worked: return table
        return sample_fc
    except ee.ee_exception.EEException:
        print(f"could not read asset {asset_name} (probably not created yet)")
        return None


def get_or_create_worldwide_sample_points(seed):
    asset_name = f'{base_asset_directory}/samples{num_samples}_seed{seed}'
    sample_fc = read_sample(asset_name)
    if sample_fc:
        return sample_fc

    print(f"creating sample {asset_name}")

    total_area = get_total_area()

    def sample_region(region_fc):
        total_sample_points = ee.Number(region_fc.aggregate_sum('sampleSize'))
        sample_image = ee.Image(1)\
            .clipToCollection(region_fc)
        sampled_region = sample_image\
            .sample(
                region=region_fc,
                numPixels=total_sample_points,
                projection=model_projection,
                scale=model_scale,
                geometries=True,
                seed=seed
            )
        return sampled_region

    def set_num_samples_to_region(region_name):
        def set_num_samples_to_take(feature):
            region_sample_size = ee.Number(feature.geometry().area()).divide(total_area).multiply(num_samples).floor()
            return feature.set('sampleSize', region_sample_size)

        region_fc = region_boundaries(region_name)
        region_fc_with_sample_size = region_fc.map(lambda feature: set_num_samples_to_take(feature))
        return region_fc_with_sample_size

    # find how many samples to take for each region, based on its area
    regions_with_sample_sizes = list(map(set_num_samples_to_region, world_regions))

    # take that many samples for each region
    sample_points = list(map(sample_region, regions_with_sample_sizes))

    # create a single region to export
    sample_fc = ee.FeatureCollection(sample_points).flatten()

    task = ee.batch.Export.table.toAsset(
        collection=sample_fc,
        assetId=asset_name,
        description=asset_name.replace('/', '_')
    )
    task.start()
    wait_for_task_completion([task], exit_if_failures=True)
    return read_sample(asset_name)


def main():
    ee.Initialize()
    get_or_create_worldwide_sample_points(train_seed)
    export_asset_table_to_drive(f'{base_asset_directory}/samples_{num_samples}')


if __name__ == '__main__':
    main()
