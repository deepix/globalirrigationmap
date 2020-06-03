# Classifies features on any given year and creates results table
# Requires: features already available as a single GEE asset
# Note: .getInfo() calls are blocking

import ee

from common import (model_scale, wait_for_task_completion, get_selected_features_image, model_snapshot_path_prefix,
                    get_selected_features, get_binary_labels, model_projection, num_samples)
from sampler import get_worldwide_sample_points


def assess_model(classifier, test_partition):
    validated = test_partition.classify(classifier)

    # Get a confusion matrix representing expected accuracy.
    if classifier.mode() != 'PROBABILITY':
        validation_matrix = validated.errorMatrix('TLABEL', 'classification')
        print('Validation error matrix: ', validation_matrix.getInfo())
        print('Validation accuracy: ', validation_matrix.accuracy().getInfo())
        print('Validation kappa: ', validation_matrix.kappa().getInfo())


def train_model(training_partition, feature_list):
    # same as in R (ntree)
    num_trees = 1000
    # same as in R (sampsize)
    bag_fraction = 0.63
    # derived from tuning in R (mtry)
    variables_per_split = 8

    classifier = ee.Classifier.randomForest(
        numberOfTrees=num_trees,
        bagFraction=bag_fraction,
        variablesPerSplit=variables_per_split,
        seed=10
    )
    classifier = classifier.train(
        features=training_partition,
        classProperty='TLABEL',
        inputProperties=feature_list,
        subsamplingSeed=10
    )

    # Model times out: uncomment if you like
    # confusion_matrix = classifier.confusionMatrix()
    # print('Training confusion matrix: ', confusion_matrix.getInfo())
    # print('Training accuracy: ', confusion_matrix.accuracy().getInfo())
    # print('Training kappa: ', confusion_matrix.kappa().getInfo())
    return classifier


def prepare_classifier_input(features_image, labels_image, sample_points):
    classifier_input = features_image
    # Labels may or may not be present (train vs. predict)
    if labels_image:
        classifier_input = ee.Image.cat(features_image, labels_image)

    classifier_input_samples = classifier_input.sampleRegions(
            collection=sample_points,
            projection=model_projection,
            scale=model_scale,
            geometries=True
        )

    prepared_input_dict = dict(
        classifier_input_samples=classifier_input_samples,
        feature_list=features_image.bandNames()
    )
    return prepared_input_dict


def create_classifier(features_image, labels_image, sample_points):
    def train_test_split(data_fc):
        split = 0.9  # Same as in R (0.9 of data for training with 5-fold cross-validation, 0.1 held out as test)
        with_random = data_fc.randomColumn('random', 10)
        train_partition = with_random.filter(ee.Filter.lt('random', split))
        test_partition = with_random.filter(ee.Filter.gte('random', split))
        return dict(training_partition=train_partition, test_partition=test_partition)

    ci = prepare_classifier_input(features_image, labels_image, sample_points)
    split = train_test_split(ci['classifier_input_samples'])
    classifier = train_model(split['training_partition'], ci['feature_list'])
    # Model times out: uncomment if you like
    # if labels_image is not None:
    #     assess_model(classifier, split['test_partition'])
    # GEE doesn't allow us to save a model so we always train the model
    # classifier = classifier.setOutputMode('PROBABILITY')
    # classifier = train_model(split['training_partition'], feature_list)
    return classifier


def build_worldwide_model():
    sample_points = get_worldwide_sample_points()
    training_image = ee.Image(f"{model_snapshot_path_prefix}_training_sample{num_samples}_all_features_labels_image")
    features_list = get_selected_features()
    features_image = training_image.select(features_list)
    labels_image = training_image.select("TLABEL")
    classifier = create_classifier(features_image, labels_image, sample_points)
    return classifier


def classify_year(classifier, model_year):
    asset_description = f'results_{model_year}'
    asset_name = f'{model_snapshot_path_prefix}_{asset_description}'
    features_image = get_selected_features_image(model_year)
    classified_image = features_image.classify(classifier)
    task = ee.batch.Export.image.toAsset(
        image=classified_image,
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
    classifier = build_worldwide_model()
    model_years = ['2000'] # , '2003', '2006', '2009', '2012', '2015', '2018']
    tasks = []
    for year in model_years:
        task = classify_year(classifier, year)
        tasks.append(task)
    wait_for_task_completion(tasks)


if __name__ == '__main__':
    main()
