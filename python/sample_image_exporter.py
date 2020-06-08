import ee

from common import model_scale, wait_for_task_completion


def export_point(point):
    TOA_BANDS = ['B3', 'B2', 'B1']
    TOA_MIN = 0.0
    TOA_MAX = 120.0
    LANDSAT_RES = 30
    RED_RGB = "#FF0000"
    RED_RGB_TRANSPARENT = RED_RGB + "00"
    sat_image = ee.Image("LANDSAT/LE7_TOA_1YEAR/2005").select(TOA_BANDS)
    point_geom = point.geometry()
    square = point_geom.buffer(model_scale).bounds()
    outer_square = point_geom.buffer(model_scale * 2).bounds()
    border_fc = ee.FeatureCollection(square)\
        .style(color=RED_RGB, fillColor=RED_RGB_TRANSPARENT)
    clipped_sat_image = sat_image \
        .clipToCollection(ee.FeatureCollection(outer_square)) \
        .visualize(bands=TOA_BANDS, min=TOA_MIN, max=TOA_MAX) \
        .blend(border_fc)
    lonlat = point_geom.coordinates().getInfo()
    prefix = f"{lonlat[0]}_{lonlat[1]}_"
    task = ee.batch.Export.image.toDrive(clipped_sat_image, folder="testImage2", scale=LANDSAT_RES,
                                         fileNamePrefix=prefix, region=outer_square)
    task.start()
    return task


if __name__ == '__main__':
    ee.Initialize()
    table = ee.FeatureCollection("users/deepakna/w210_irrigated_croplands/FPValidationSamples")
    f = table.first()
    t = export_point(f)
    wait_for_task_completion([t])
