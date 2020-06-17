import ee

from common import model_scale, wait_for_task_completion, model_projection, base_asset_directory


def export_point(id: str, coords: list, folder: str) -> ee.batch.Task:
    print(f"point: {id}, {coords}")
    TOA_BANDS = ['B3', 'B2', 'B1']
    TOA_MIN = 0.0
    TOA_MAX = 120.0
    LANDSAT_RES = 30
    RED_RGB = "#FF0000"
    RED_RGB_TRANSPARENT = RED_RGB + "00"
    sat_image = ee.Image("LANDSAT/LE7_TOA_1YEAR/2005").select(TOA_BANDS)
    point_geom = ee.Geometry.Point(coords=coords, proj=model_projection)
    square = point_geom.buffer(model_scale).bounds()
    outer_square = point_geom.buffer(model_scale * 2).bounds()
    border_fc = ee.FeatureCollection(square)\
        .style(color=RED_RGB, fillColor=RED_RGB_TRANSPARENT)
    clipped_sat_image = sat_image \
        .clipToCollection(ee.FeatureCollection(outer_square)) \
        .visualize(bands=TOA_BANDS, min=TOA_MIN, max=TOA_MAX) \
        .blend(border_fc)
    prefix = f"{id}"
    task = ee.batch.Export.image.toDrive(clipped_sat_image, folder=folder, scale=LANDSAT_RES,
                                         fileNamePrefix=prefix, region=outer_square)
    task.start()
    return task


def export_samples(table_name: str) -> None:
    table = ee.FeatureCollection(f"{base_asset_directory}/{table_name}").getInfo()
    tasks = list(map(
        lambda p: export_point(p['id'], p['geometry']['coordinates'], table_name),
        table['features']
    ))
    wait_for_task_completion(tasks)


if __name__ == '__main__':
    ee.Initialize()
#    export_samples("FNValidationSamplesv3")
    export_samples("TLabelv2Validation")
