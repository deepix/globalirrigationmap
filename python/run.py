import ee
import common
import features_exporter
import classifier as clf


def main(model_years):
    ee.Initialize()
    classifier = clf.build_worldwide_model()
    for year in model_years:
        tasks = []
        task = features_exporter.export_selected_features_for_year(year)
        tasks.append(task)
        common.wait_for_task_completion(tasks)
        tasks = []
        task = clf.classify_year(classifier, year)
        tasks.append(task)
        common.wait_for_task_completion(tasks)


if __name__ == '__main__':
    years = ["2000", "2003", "2006", "2009", "2012", "2015", "2018"]
    main(years)
