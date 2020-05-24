import time
import itertools
import datetime

import ee

# Checklist when changing model:
# 1. Update model_snapshot_version below
# 2. Change selectedBands in dataset_list[] if your feature list is different
# 3. Change get_binary_labels() below if your label threshold has changed
# 4. Change model hyperparameters in classifier.py if they are different
# 5. Set num_samples in classifier.py:build_worldwide_model() if you want to use a different number of samples

# v1: use land cover feature
# v2a: use EVI amplitude etc from MCD12Q2.006
# v2b: use land cover only but from collection 6 of MODIS
model_snapshot_version = "post_mids_v2b"
# Create this directory ahead of time
base_asset_directory = "users/deepakna/w210_irrigated_croplands"
#
model_snapshot_path_prefix = f"{base_asset_directory}/{model_snapshot_version}"
model_projection = "EPSG:4326"

# CONFIGURATION flags
label_type = "MIRCA2K"  # or GFSAD1000

dataset_list = [
    {
        'datasetLabel': "MODIS/MOD09GA_NDWI",
        'allBands': ["NDWI"],
        'selectedBands': [],
        'summarizer': "max"
    },
    {
        'datasetLabel': "IDAHO_EPSCOR/TERRACLIMATE",
        'allBands': ["aet", "def", "pdsi", "pet", 'pr', 'soil', "srad", "swe", "tmmn", 'tmmx', 'vap', 'vpd', 'vs'],
        # v1
        'selectedBands': ["tmmx", "pet", "vpd"],
        'summarizer': "mean",
        'missingValues': -9999
    },
    {
        'datasetLabel': "NASA/GRACE/MASS_GRIDS/LAND",
        'allBands': ["lwe_thickness_jpl"],
        'selectedBands': [],
        'summarizer': "mean"
    },
    {
        'datasetLabel': "NASA/GLDAS/V021/NOAH/G025/T3H",
        'allBands': ["Albedo_inst", "AvgSurfT_inst", "CanopInt_inst", "ECanop_tavg", "ESoil_tavg", "Evap_tavg",
                  "LWdown_f_tavg", "Lwnet_tavg", "PotEvap_tavg", "Psurf_f_inst", "Qair_f_inst", "Qg_tavg", "Qh_tavg",
                  "Qle_tavg", "Qs_acc", "Qsb_acc", "Qsm_acc", "Rainf_f_tavg", "Rainf_tavg", "RootMoist_inst",
                  "SWE_inst", "SWdown_f_tavg", "SnowDepth_inst", "Snowf_tavg", "SoilMoi0_10cm_inst",
                  "SoilMoi10_40cm_inst", "SoilMoi100_200cm_inst", "SoilMoi40_100cm_inst", "SoilTMP0_10cm_inst",
                  "SoilTMP10_40cm_inst", "SoilTMP100_200cm_inst", "SoilTMP40_100cm_inst", "Swnet_tavg", "Tair_f_inst",
                  "Tveg_tavg", "Wind_f_inst"],
        # v1
        'selectedBands': ["Albedo_inst", "Tveg_tavg"],
        'summarizer': "mean",
        'missingValues': -9999
    },
    {
        'datasetLabel': "MODIS/006/MOD13A2",
        'allBands': ["NDVI", "EVI"],
        'selectedBands': ["NDVI", "EVI"],
        'summarizer': "max",
        'missingValues': -9999
    },
    {
        'datasetLabel': "MODIS/006/MCD12Q1",
        'allBands': ["LC_Type1", "LC_Type2"],
        'selectedBands': ["LC_Type1", "LC_Type2"],
        'summarizer': "max",
        'minYear': '2001',
        'missingValues': -9999
    },
    {
        'datasetLabel': 'MODIS/006/MCD12Q2',
        'allBands': ['NumCycles', "EVI_Minimum_1", "EVI_Minimum_2", "EVI_Amplitude_1", "EVI_Amplitude_2"],
        'allDateBands': ["Greenup_1", "Greenup_2", "MidGreenup_1", "MidGreenup_2", "Peak_1", "Peak_2",
                         "MidGreendown_1", "MidGreendown_2", "Senescence_1", "Senescence_2"],
        'selectedBands': ["EVI_Amplitude_1"],
        'summarizer': "mean",
        'minYear': '2001',
        'maxYear': '2017',
        'missingValues': -9999
    }
]

# We split world regions into 2 to avoid exceeding GEE geometry limits
# Error: Geometry has too many edges (3970390 > 2000000)
world_regions_1 = [
    "North America",
    "Central America",
    "South America",
    "Australia",
    "Africa",
    # This causes geometry explosion, not including
    # "Oceania",
]
world_regions_2 = [
    "Europe",
    "SW Asia",
    "Central Asia",
    "N Asia",
    "E Asia",
    "SE Asia",
    "S Asia",
    "Caribbean"
]
model_regions = [
    "world1",
    "world2"
]
model_scale = 8000


def get_features_from_dataset(dataset, which, model_year, region_fc):
    assert which in ['all', 'selected'], "Specify which bands to get: all or selected"

    if which == 'all':
        features = dataset['allBands']
    elif which == 'selected':
        features = dataset['selectedBands']
    else:
        raise NotImplementedError("Specify which bands to get: all or selected")

    if not features:
        return None

    image = get_features_image_from_dataset(dataset, features, model_year, region_fc)

    # Clumsy logic to handle onset days.  It should have been days since start of year,
    # but they use days since start of epoch (1970-01-01).  We convert it back into
    # days since start of year.
    if 'allDateBands' in dataset and which == 'all':
        date_features = dataset['allDateBands']
        image2 = get_features_image_from_dataset(dataset, date_features, model_year, region_fc)
        days_since_epoch = (datetime.datetime(year=int(model_year), month=1, day=1) -
                            datetime.datetime(year=1970, month=1, day=1)).days
        new_bands = list(map(lambda b: image2.select(b).expression(f'b(0) - {days_since_epoch}'), date_features))
        date_bands_image = ee.Image.cat(*new_bands)
        image = image.addBands(date_bands_image)
    if 'missingValues' in dataset:
        mask_image = ee.Image(dataset['missingValues']).clipToCollection(region_fc)
        image = image.unmask(mask_image)
    return image


def get_features_image_from_dataset(dataset, features, model_year, region_fc):
    data_source = dataset['datasetLabel']
    if 'minYear' in dataset and int(model_year) < int(dataset['minYear']):
        new_model_year = int(dataset['minYear'])
        print(f"Warning: model year {model_year} is earlier than dataset {data_source}, using {new_model_year} instead")
        model_year = str(new_model_year)
    if 'maxYear' in dataset and int(model_year) > int(dataset['maxYear']):
        new_model_year = int(dataset['maxYear'])
        print(f"Warning: model year {model_year} is later than dataset {data_source}, using {new_model_year} instead")
        model_year = str(new_model_year)
    image_collection = ee.ImageCollection(data_source) \
        .select(features) \
        .filterDate(model_year + "-01-01", model_year + "-12-31") \
        .map(lambda img: img.clipToCollection(region_fc))

    if dataset['summarizer'] == "mean":
        image = image_collection \
            .mean()
    elif dataset['summarizer'] == "max":
        image = image_collection \
            .max()
    else:
        raise ValueError("unknown summarizer")
    return image


def region_boundaries(region):
    # we assume 2-characters = country FIPS code
    if len(region) == 2:
        fc = ee \
            .FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
            .filterMetadata("country_co", "equals", region)
    elif region == "world1":
        fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
            .filter(ee.Filter.inList("wld_rgn", ee.List(world_regions_1)))
    elif region == "world2":
        fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
            .filter(ee.Filter.Or(
                ee.Filter.inList("wld_rgn", ee.List(world_regions_2)),
                # include 2 relatively big countries in Oceania region
                ee.Filter.inList("country_co", ee.List(["NZ", "PP"]))
            ))
    else:
        fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
            .filterMetadata("wld_rgn", "equals", region)
    return fc.map(lambda f: f.set("areaHa", f.geometry().area()))


def wait_for_task_completion(tasks, exit_if_failures=False):
    sleep_time = 10  # seconds
    done = False
    failed_tasks = []
    while not done:
        failed = 0
        completed = 0
        for t in tasks:
            status = t.status()
            print(f"{status['description']}: {status['state']}")
            if status['state'] == 'COMPLETED':
                completed += 1
            elif status['state'] in ['FAILED', 'CANCELLED']:
                failed += 1
                failed_tasks.append(status)
        if completed + failed == len(tasks):
            print(f"All tasks processed in batch: {completed} completed, {failed} failed")
            done = True
        time.sleep(sleep_time)
    if failed_tasks:
        print("--- Summary: following tasks failed ---")
        for status in failed_tasks:
            print(status)
        print("--- END Summary ---")
        if exit_if_failures:
            raise NotImplementedError("There were some failed tasks, see report above")


def get_features_image(region_fc, model_year, which):
    # get all data in parallel, and build a list of images
    # there's a lot to unpack in the code below, but think of it as a loop on the datasets above
    images = list(itertools.chain.from_iterable(map(
        lambda dataset: [get_features_from_dataset(dataset, which, model_year, region_fc)], dataset_list
    )))
    return images


def get_labels(region_fc):
    if label_type == "MIRCA2K":
        label_image = ee.Image(f"{base_asset_directory}/mc4MaxIrrigatedHaLabels") \
            .clipToCollection(region_fc)
        return label_image \
            .select(['b1'], ['IRRIGATED']) \
            .expression('LABEL = b("IRRIGATED")')
    elif label_type == "GFSAD1000":
        label_image = ee.Image('USGS/GFSAD1000_V0') \
            .select('landcover') \
            .expression('LABEL = (b(0) > 0 && b(0) < 4) ? 1 : 0')
        return label_image
    else:
        raise NotImplementedError("Unknown label type")


def get_binary_labels(label_image):
    # Cut-off derived from offline model analysis
    return label_image \
        .expression('BLABEL = (log(b("LABEL") + 1) >= 1 ? 1 : 0)')


def get_selected_features():
    selected_features = ['X', 'Y']
    for ds in dataset_list:
        selected_features.extend(ds['selectedBands'])
    return selected_features


def get_selected_features_image(model_year):
    def get_selected_features_in_region(region):
        region_fc = region_boundaries(region)
        features_image = get_features_image(region_fc, model_year, 'selected')
        lonlat_image = ee.Image.pixelLonLat() \
            .clipToCollection(region_fc) \
            .select(['longitude', 'latitude'], ['X', 'Y'])
        features_with_lonlat_image = ee.Image.cat(features_image, lonlat_image)
        return features_with_lonlat_image

    regions = ["world1", "world2"]
    regional_features = list(map(get_selected_features_in_region, regions))
    assert (len(regions) == 2)
    one_image = ee.Image(regional_features[0]).blend(regional_features[1])
    return one_image


def export_asset_image_to_drive(asset_id):
    image = ee.Image(asset_id)
    folder = asset_id.replace('/', '_')
    print(f"Downloading to gdrive: {folder}")
    task = ee.batch.Export.image.toDrive(
        image=image.toFloat(),
        folder=folder,
        description=asset_id.replace('/', '_'),
        crs=model_projection,
        scale=model_scale
    )
    task.start()
    wait_for_task_completion([task], True)


def export_asset_table_to_drive(asset_id):
    fc = ee.FeatureCollection(asset_id)
    folder = asset_id.replace('/', '_')
    max_len = 100
    if len(folder) >= max_len:
        folder = folder[:max_len]
        print(f"folder length is too long (truncating to {max_len})")
    print(f"Downloading table {asset_id} to gdrive: {folder}")
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        folder=folder,
        description=folder,
        fileFormat='GeoJSON'
    )
    task.start()
    wait_for_task_completion([task], True)


def sample_to_image(sample_points, properties):
    images = list(map(lambda prop: sample_points
        .filter(ee.Filter.notNull([prop]))
        .reduceToImage([prop], ee.Reducer.first())
        .select(['first'], [prop]),
        properties
    ))
    return ee.Image.cat(*images).select(properties, get_selected_features())


def export_sample_to_image():
    sample_points = ee.FeatureCollection("users/deepakna/smote_training_sample_table_10000")
    properties = ['X', 'Y', 'tmmx', 'vap', 'vpd', 'Albd_ns', 'ECnp_tv', 'RtMst_n', 'SWdwn__', 'Tvg_tvg', 'NDVI']
    sample_image = sample_to_image(sample_points, properties)
    task = ee.batch.Export.image.toAsset(
        image=sample_image,
        description="SampleImageExport",
        assetId="users/deepakna/smote_training_sample_image_10000",
        crs=model_projection,
        scale=model_scale
    )
    task.start()
    wait_for_task_completion([task], True)
