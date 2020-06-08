import ee

from sampler import get_or_create_worldwide_sample_points
from common import base_asset_directory, assess_seed


def assess_map(map_image):
    sample_points = get_or_create_worldwide_sample_points(assess_seed)
    sampled_region = map_image \
        .reduceRegions(collection=sample_points, reducer=ee.Reducer.first().forEachBand(map_image)) \
        .map(lambda f: f.select(['actual', 'pred']))
    confusion_matrix = sampled_region.errorMatrix(actual="actual", predicted="pred")
    print(f"Confusion matrix: {confusion_matrix.getInfo()}")
    print(f"Kappa: {confusion_matrix.kappa().getInfo()}")
    print(f"Accuracy: {confusion_matrix.accuracy().getInfo()}")
    print("-----")


def assess_cropland_model_only():
    cropland_map = ee.Image("users/deepakna/ellecp/2005_ternary") \
        .addBands(ee.Image(f"{base_asset_directory}/s2005tlabels")) \
        .select(["b1", "TLABEL"], ["pred", "actual"])
    print("Cropland model assessment (Elle)")
    cl_mask = ee.Image(f'{base_asset_directory}/CLMask')
    assess_map(cropland_map.mask(cl_mask))


def assess_timestationary_model_only():
    ts_map = ee.Image(f"{base_asset_directory}/post_mids_v3b_results_2005") \
        .addBands(ee.Image(f"{base_asset_directory}/s2005tlabels")) \
        .select(["classification", "TLABEL"], ["pred", "actual"])
    print("Time-stationary model assessment (Deepak)")
    assess_map(ts_map)

def assess_model_results():
    final_map = ee.Image(f'{base_asset_directory}/s2005AssessmentMap')
    assess_cropland_model_only()
    assess_timestationary_model_only()
    print("Combined model assessment")
    assess_map(final_map)


if __name__ == '__main__':
    ee.Initialize()
    assess_model_results()
