# image_clustering
The project offers methods for image clustering.
It is based on features extracted from Neural Networks (support both deep and shallow), and cluster images using KMeans Algorithm.

### Usage
Image Clustering can be used with a command line argument:
python cluster.py [-args]
For information on args, use "python cluster.py -h"
Example:
python cluster.py -extractor=autoencoder -input_dir=$INPUT_DIR -output_dir=$OUTPUT_DIR -model_path=$MODEL_PATH
python cluster.py -extractor=autoencoder -input_dir=$INPUT_DIR -output_dir=$OUTPUT_DIR -train
python cluster.py -extractor=detectionnetwork -input_dir=$INPUT_DIR -output_dir=$OUTPUT_DIR -base_model=vgg16_bn

### Feature Extractors
Feature Extractors are defined under extractors/.
To add a feature extractor, 
1. add the file to extractors.
2. create a subclass of FeatureExtractor in the file.
3. Implement abstract functions
4. define 'type' which can be used as an input on command line to use this extractor.
5. You are done

