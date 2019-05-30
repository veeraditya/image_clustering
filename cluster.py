from helpers import *
from argparse import ArgumentParser
import torch


def main() -> None:
    """
    Main runner function

    Returns:
        None
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-extractor", "--extractor", type=str, required=True,
                            help="feature extractor to be used: autoencoder or detectionnetwork")
    arg_parser.add_argument("-number_of_clusters", "--number_of_clusters", type=int, required=False, default=10,
                            help="number of clusters desired")
    arg_parser.add_argument("-output_dir", "--output_dir", type=str, required=True,
                            help="directory path to be used to output cluster and model if trained")
    arg_parser.add_argument("-input_dir", "--input_dir", type=str, required=True,
                            help="directory path to the input folder containing all images")
    arg_parser.add_argument("-base_model", "--base_model", type=str, required=False, default="vgg16_bn")
    arg_parser.add_argument("-model_path", "--model_path", type=str, required=False,
                            help="path to the AutoEncoder model file attached along")
    arg_parser.add_argument("-train", "--train", required=False, default=False, action='store_true',
                            help="if set along with autoencoder choice, will train the network on images before using")
    arg_parser.add_argument("-epochs", "--epochs", type=int, required=False, default=500,
                            help="number of epochs for training, if training")
    arg_parser.add_argument("-milestone_step_ratio", "--milestone_step_ratio",
                            type=list, required=False, default=[0.2, 0.8],
                            help="steps for switching learning rate defined as a fraction, of number of epochs")
    arg_parser.add_argument("-batch_size", "--batch_size", type=int, default=64, required=False,
                            help="batch size for training")
    arg_parser.add_argument("-learning_rate", "--learning_rate", required=False, default=0.01,
                            help="learning rate for training AutoEncoder")
    arg_parser.add_argument("-weight_decay", "--weight_decay", required=False, default=0,
                            help="weight decay for training")
    arg_parser.add_argument("-gamma", "--gamma", required=False, default=0.1,
                            help="Gamma used for tuning learning rate while training")
    arg_parser.add_argument("-evaluate_metric", "--evaluate_metric", required=False, default=False, action='store_true',
                            help="whether the code should report silhouette clustering score for clustering")

    params = vars(arg_parser.parse_args())

    params['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['input_shape'] = (224, 224, 3)

    output_path = params['output_dir']

    make_directories(output_path)
    if params['train']:
        params['input_train'] = os.path.join(params['output_dir'], 'input_training')
        shutil.copytree(params['input_dir'], os.path.join(params['input_train'], '0'))

    feature_extractors = get_feature_extractors()
    if params['extractor'] not in feature_extractors.keys():
        print('invalid feature extractor. choose either of two: ' + str(list(feature_extractors.keys())))
    feature_extractor = feature_extractors[params['extractor']](params)
    perform_clustering(output_path, feature_extractor, params)


if __name__ == '__main__':
    main()
