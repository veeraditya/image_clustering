from tqdm._tqdm import tqdm
import os, glob, shutil
from extractors.base import FeatureExtractor
import extractors
from PIL import Image
import numpy as np

from sklearn.cluster import KMeans
import sklearn.metrics as metrics


def make_directories(output_path: str):
    shutil.rmtree(output_path, ignore_errors=True)
    for directory in ['model', 'clusters']:
        directory_path = os.path.join(output_path, directory)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)


def create_feature_map(img_map: dict, model: FeatureExtractor):
    feature_map = {}
    for img_index, img_path in tqdm(list(img_map.items()), desc="Extracting features.."):
        img = Image.open(img_path)
        feature_map[img_index] = model.extract_features(img)
    return feature_map


def get_input_image_map(input_folder: str):
    input_folder = input_folder.rstrip(os.sep)
    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))
    img_map = {k: v for k, v in enumerate(image_files)}
    return img_map


def get_feature_extractors()->dict:
    all_extractors = list(extractors.iter_extractor_clss())
    all_extractors = {name: extractor for name, extractor in all_extractors}
    return all_extractors


def perform_clustering(output_path: str, extractor: FeatureExtractor, params):
    input_dir, num_desired_clusters = params['input_dir'], params['number_of_clusters']
    image_map = get_input_image_map(input_dir)
    feature_map = create_feature_map(image_map, extractor)
    print("Clustering..")
    clusters = KMeans(n_clusters=num_desired_clusters, n_init=10).fit(list(feature_map.values()))
    move_files_to_clusters(output_path, num_desired_clusters, clusters, image_map)
    if params['evaluate_metric']:
        print("clustering score: ", evaluate_clustering(np.array(list(feature_map.values())), clusters.labels_))


def move_files_to_clusters(output_path, num_desired_clusters, clusters, img_map):
    shutil.rmtree(os.path.join(output_path, 'clusters'))
    for cluster_label in range(num_desired_clusters):
        cluster_path = os.path.join(output_path, 'clusters', str(cluster_label))
        os.makedirs(cluster_path, exist_ok=True)
        img_idcs = np.where(clusters.labels_ == cluster_label)
        img_paths = np.array(list(img_map.values()))[img_idcs]
        for img in img_paths:
            shutil.copy(img, cluster_path)
    print('moved files to cluster folder: ', output_path)


def evaluate_clustering(feature_map, cluster_labels):
    return metrics.silhouette_score(feature_map, cluster_labels)