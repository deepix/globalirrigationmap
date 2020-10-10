import ee

from common import model_scale, wait_for_task_completion, model_projection
from common import train_seed, label_path
from sampler import get_or_create_worldwide_sample_points

TOA_BANDS = ['B3', 'B2', 'B1']
TOA_MIN = 0.0
TOA_MAX = 120.0
LANDSAT_RES = 30


def export_point_unbuffered(id: str, coords: list, folder: str) -> ee.batch.Task:
    print(f"point: {id}, {coords}")
    sat_image = ee.Image("LANDSAT/LE7_TOA_1YEAR/2005").select(TOA_BANDS)
    point_geom = ee.Geometry.Point(coords=coords, proj=model_projection)
    square = point_geom.buffer(model_scale).bounds()
    clipped_sat_image = sat_image \
        .clipToCollection(ee.FeatureCollection(square)) \
        .visualize(bands=TOA_BANDS, min=TOA_MIN, max=TOA_MAX)
    prefix = f"{id}"
    task = ee.batch.Export.image.toDrive(clipped_sat_image, folder=folder, scale=LANDSAT_RES,
                                         fileNamePrefix=prefix, region=square)
    task.start()
    return task


# get labels image
if __name__ == '__main__':
    num_pictures = 3000   # 5 times for class labels 1 or 2

    ee.Initialize()
    sample_points_fc = get_or_create_worldwide_sample_points(train_seed)
    labels_image = ee.Image(label_path) \
        .expression(f"TLABEL = (b(0) == 0 ? 1 : 0)")
    labels_image = labels_image.mask(labels_image)
    labels_fc = labels_image \
        .sample(
            region=sample_points_fc,
            numPixels=num_pictures, # it seems to get a few more than this, and GEE limit is 3K
            scale=model_scale,
            projection=model_projection,
            seed=train_seed,
            geometries=True,
            dropNulls=True
        )
    print(labels_fc.size().getInfo())

    labels_fc_info = labels_fc.getInfo()
    tasks = list(map(
        lambda p: export_point_unbuffered(p['id'], p['geometry']['coordinates'], "classNotIrr_samples"),
        labels_fc_info['features']
    ))
    wait_for_task_completion(tasks)
