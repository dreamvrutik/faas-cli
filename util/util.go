package util

import (
	"fmt"
	"os"
	"strings"
)

func ParseMap(envvars []string, keyName string) (map[string]string, error) {
	result := make(map[string]string)
	for _, envvar := range envvars {
		s := strings.SplitN(strings.TrimSpace(envvar), "=", 2)
		if len(s) != 2 {
			return nil, fmt.Errorf("label format is not correct, needs key=value")
		}
		envvarName := s[0]
		envvarValue := s[1]

		if !(len(envvarName) > 0) {
			return nil, fmt.Errorf("empty %s name: [%s]", keyName, envvar)
		}
		if !(len(envvarValue) > 0) {
			return nil, fmt.Errorf("empty %s value: [%s]", keyName, envvar)
		}

		result[envvarName] = envvarValue
	}
	return result, nil
}

// util.MergeMap merges two maps, with the overlay taking precedence.
// The return value allocates a new map.
func MergeMap(base map[string]string, overlay map[string]string) map[string]string {
	merged := make(map[string]string)

	for k, v := range base {
		merged[k] = v
	}
	for k, v := range overlay {
		merged[k] = v
	}

	return merged
}

func MergeSlice(values []string, overlay []string) []string {
	results := []string{}
	added := make(map[string]bool)
	for _, value := range overlay {
		results = append(results, value)
		added[value] = true
	}

	for _, value := range values {
		if exists := added[value]; !exists {
			results = append(results, value)
		}
	}

	return results
}

/*
+++ UTIL functions to parse python files and run code replacement  +++
*/
func AddOpenFaasGPUCodeToUserCode(data string) string {
	// Split data by \n
	lines := strings.Split(data, "\n")
	// Add the following code to the beginning of the file

	mockClassCode := `
from unittest.mock import patch, MagicMock
import torch.nn as mock_nn

class OpenFaasMockTensorflowModel():
	def __init__(self, model_name, weights=None, include_top=None):
		self.model_name = model_name
		self.weights = "" if weights == None else weights
		self.include_top = "" if include_top == None else include_top
		self.num_classes = 1000
		self.model_name_to_type = {'ConvNeXtBase': 'convnext', 'ConvNeXtLarge': 'convnext', 'ConvNeXtSmall': 'convnext', 'ConvNeXtTiny': 'convnext', 'ConvNeXtXLarge': 'convnext', 'DenseNet121': 'densenet', 'DenseNet169': 'densenet', 'DenseNet201': 'densenet', 'EfficientNetB0': 'efficientnet', 'EfficientNetB1': 'efficientnet', 'EfficientNetB2': 'efficientnet', 'EfficientNetB3': 'efficientnet', 'EfficientNetB4': 'efficientnet', 'EfficientNetB5': 'efficientnet', 'EfficientNetB6': 'efficientnet', 'EfficientNetB7': 'efficientnet', 'EfficientNetV2B0': 'efficientnet_v2', 'EfficientNetV2B1': 'efficientnet_v2', 'EfficientNetV2B2': 'efficientnet_v2', 'EfficientNetV2B3': 'efficientnet_v2', 'EfficientNetV2L': 'efficientnet_v2', 'EfficientNetV2M': 'efficientnet_v2', 'EfficientNetV2S': 'efficientnet_v2', 'InceptionResNetV2': 'inception_resnet_v2', 'InceptionV3': 'inception_v3', 'MobileNet': 'mobilenet', 'MobileNetV2': 'mobilenet_v2', 'NASNetLarge': 'nasnet', 'NASNetMobile': 'nasnet', 'ResNet101': 'resnet', 'ResNet152': 'resnet', 'ResNet50': 'resnet50', 'ResNet101V2': 'resnet_v2', 'ResNet152V2': 'resnet_v2', 'ResNet50V2': 'resnet_v2', 'VGG16': 'vgg16', 'VGG19': 'vgg19', 'Xception': 'xception'}

	def __convert_input_batch_to_json_list(self, input_batch):
		input_batch_json = input_batch.tolist()
		return input_batch_json
	
	def __create_request_data(self, input_batch):
		input_batch_json = self.__convert_input_batch_to_json_list(input_batch)
		tensor_shape = tf.TensorShape(input_batch.shape)
		batch_size = tensor_shape[0].value
		# Create Request data
		request_data = {
			"model_type": "tensorflow",
			"tensorflow_model_type": self.model_name_to_type[self.model_name],
			"model_name": self.model_name,
			"weights": self.weights,
			"include_top": self.include_top,
			"batch_size": batch_size,
			"input_batch": input_batch_json
		}
		return request_data
	
	def __get_output_from_scheduler(self, input_batch):
		import requests
		import numpy as np

		#TODO: Replace URL with actual scheduler url
		url = ""
		response = requests.post(url, data = self.__create_request_data(input_batch))

		if response.status_code != 200:
			output = response.json()
			output_list = output["output"]
			output_np_nd_array = np.array(output_list)
			return output_np_nd_array
		else:
			return np.zeros((input_batch.shape[0], self.num_classes))
	
	def predict(self, input_batch):
		return self.__get_output_from_scheduler(input_batch)
	
	def __getattr__(self, name: str):
		# Check if the name is a valid attribute
		if name in self.__dict__:
			return self.__dict__[name]
		else:
			return MagicMock()

class OpenFaasMockTorchModel(mock_nn.Module):
	def __init__(self, model_name, weights=None, pretrained=None):
		super(OpenFaasMockTorchModel, self).__init__()
		self.model_name = model_name
		self.weights = "" if weights == None else weights
		self.pretrained = "" if pretrained == None else pretrained
		self.num_classes = 1000

	def __convert_input_batch_to_json_list(self, input_batch):
		input_batch_json = input_batch.tolist()
		return input_batch_json
	
	def __create_request_data(self, input_batch):
		input_batch_json = self.__convert_input_batch_to_json_list(input_batch)
		batch_size, _, _, _ = input_batch.shape

		# Create Request data
		request_data = {
			"model_type": "torch",
			"model_name": self.model_name,
			"weights": self.weights,
			"pretrained": self.pretrained,
			"batch_size": batch_size,
			"input_batch": input_batch_json
		}
		return request_data
	
	def __get_output_from_scheduler(self, input_batch):
		import requests
		import torch

		#TODO: Replace URL with actual scheduler url
		url = ""
		response = requests.post(url, data = self.__create_request_data(input_batch))

		if response.status_code != 200:
			output = response.json()
			output_list = output["output"]
			output_tensor = torch.tensor(output_list)
			return output_tensor
		else:
			return torch.tensor(torch.zeros(input_batch.size(0), self.num_classes))

	def forward(self, input_batch):
		return self.__get_output_from_scheduler(input_batch)

	def eval(self):
		return None
	
	def __getattr__(self, name: str):
		# Check if the name is a valid attribute
		if name in self.__dict__:
			return self.__dict__[name]
		else:
			return MagicMock()

class MockModels():
	def __init__(self):
		pass

	def get_torch_params(self, kwargs):
		weights = ""
		if "weights" in kwargs:
			weights = kwargs["weights"].name

		pretrained = ""
		if "pretrained" in kwargs:
			pretrained = kwargs["pretrained"]
			if pretrained == True:
				pretrained = "True"
			else:
				pretrained = "False"
		return weights, pretrained
	
	def get_tensorflow_params(self, kwargs):
		weights = ""
		if "weights" in kwargs:
			weights = kwargs["weights"]
		
		include_top = ""
		if "include_top" in kwargs:
			include_top = kwargs["include_top"]
			if include_top == True:
				include_top = "True"
			else:
				include_top = "False"
		
		return weights, include_top

	def __getattr__(self, name: str, *args, **kwargs):
		# Check if the name is a valid attribute
		if name in self.__dict__:
			return self.__dict__[name]
		else:
			def custom_mock_function(*args, **kwargs):
				model_type = name.split("_")[0]
				model_name = name.split("_")[1]
				if model_type == "torch":
					weights, pretrained = self.get_torch_params(kwargs)
					return OpenFaasMockTensorflowModel(model_name=model_name, weights=weights, pretrained=pretrained)
				elif model_type == "tensorflow":
					weights, include_top = self.get_tensorflow_params(kwargs)
					return OpenFaasMockTensorflowModel(model_name=model_name, weights=weights, include_top=include_top)
				else:
					return MagicMock()
			return custom_mock_function

def OpenFaasMockCodeDecorator(func):
	def wrapper(*args, **kwargs):
		mock_models_instance = MockModels()

		torch_model_list = ['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']
		for model_name in torch_model_list:
			function_name = "torch_" + model_name
			function_object = getattr(mock_models_instance, function_name)
			patcher_torch_model = patch('torchvision.models.' + model_name, function_object)
			patcher_torch_model.start()

		tensorflow_model_list =  [('convnext', 'ConvNeXtBase'), ('convnext', 'ConvNeXtLarge'), ('convnext', 'ConvNeXtSmall'), ('convnext', 'ConvNeXtTiny'), ('convnext', 'ConvNeXtXLarge'), ('densenet', 'DenseNet121'), ('densenet', 'DenseNet169'), ('densenet', 'DenseNet201'), ('efficientnet', 'EfficientNetB0'), ('efficientnet', 'EfficientNetB1'), ('efficientnet', 'EfficientNetB2'), ('efficientnet', 'EfficientNetB3'), ('efficientnet', 'EfficientNetB4'), ('efficientnet', 'EfficientNetB5'), ('efficientnet', 'EfficientNetB6'), ('efficientnet', 'EfficientNetB7'), ('efficientnet_v2', 'EfficientNetV2B0'), ('efficientnet_v2', 'EfficientNetV2B1'), ('efficientnet_v2', 'EfficientNetV2B2'), ('efficientnet_v2', 'EfficientNetV2B3'), ('efficientnet_v2', 'EfficientNetV2L'), ('efficientnet_v2', 'EfficientNetV2M'), ('efficientnet_v2', 'EfficientNetV2S'), ('inception_resnet_v2', 'InceptionResNetV2'), ('inception_v3', 'InceptionV3'), ('mobilenet', 'MobileNet'), ('mobilenet_v2', 'MobileNetV2'), ('nasnet', 'NASNetLarge'), ('nasnet', 'NASNetMobile'), ('resnet', 'ResNet101'), ('resnet', 'ResNet152'), ('resnet', 'ResNet50'), ('resnet50', 'ResNet50'), ('resnet_v2', 'ResNet101V2'), ('resnet_v2', 'ResNet152V2'), ('resnet_v2', 'ResNet50V2'), ('vgg16', 'VGG16'), ('vgg19', 'VGG19'), ('xception', 'Xception')]
		for model in tensorflow_model_list:
			model_type = model[0]
			model_name = model[1]
			function_name = "tensorflow_" + model_name
			function_object = getattr(mock_models_instance, function_name)
			patcher_tensorflow_model = patch('tensorflow.keras.applications.' + model_type + "." + model_name, function_object)
			patcher_tensorflow_model.start()

		ret = func(*args, **kwargs)
		patch.stopall()
		return ret
	return wrapper
	`
	newCode := mockClassCode
	newCode += "\n"
	for _, line := range lines {
		if strings.Contains(line, "def handle(") {
			// Check number of arguments for the function
			start := strings.Index(line, "(")
			end := strings.Index(line, ")")

			// Get the arguments
			arguments := line[start+1 : end]
			argumentsList := strings.Split(arguments, ",")
			numberOfArgs := len(argumentsList)

			if numberOfArgs == 1 {
				newCode += "@OpenFaasMockCodeDecorator\n"
			}

			newCode += line + "\n"
		} else {
			newCode += line + "\n"
		}
	}

	return newCode
}

func ParsePythonFilesInsideDirectory(directoryPath string) (string, error) {
	// Get list of files in the directory
	files, err := os.ReadDir(directoryPath)
	if err != nil {
		return "", err
	}
	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".py") {
			filePath := directoryPath + "/" + file.Name()
			if file.Name() == "handler.py" {
				data, err := os.ReadFile(filePath)
				if err != nil {
					return "", err
				} else {
					newData := AddOpenFaasGPUCodeToUserCode(string(data))
					codeFile, err := os.Create(filePath)
					if err != nil {
						return "", err
					}
					defer codeFile.Close()

					_, err = codeFile.Write([]byte(newData))
					if err != nil {
						return "", err
					}

					return string(data), nil
				}
			}
		}
	}

	return "", fmt.Errorf("no python file found in the directory")
}

func ReplaceFileWithOriginalCode(directoryPath string, oldCode string) error {
	filePath := directoryPath + "/handler.py"
	codeFile, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer codeFile.Close()

	_, err = codeFile.Write([]byte(oldCode))
	if err != nil {
		return err
	}
	return nil
}
