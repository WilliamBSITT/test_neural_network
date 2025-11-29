
import sys
from nn.model import NeuralNetwork

def inspect_model(filepath):
    try:
        model = NeuralNetwork.load(filepath)
        print(f"Model loaded: {type(model)}")
        
        print(f"Optimizer: {type(model._optimizer)}")
        if hasattr(model._optimizer, 'm_w'):
            print(f"Adam optimizer state present. Keys: {len(model._optimizer.m_w)}")
        
        total_params = 0
        for layer, _ in model._layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                total_params += layer.weights.size
                print(f"Layer weights shape: {layer.weights.shape}, dtype: {layer.weights.dtype}")
            if hasattr(layer, 'bias') and layer.bias is not None:
                total_params += layer.bias.size
        
        print(f"Total parameters: {total_params}")
        
        # Check for gradients and other attributes
        has_grads = False
        for layer, _ in model._layers:
            if layer.grad_weights is not None or layer.grad_bias is not None:
                has_grads = True
                break
        print(f"Gradients present: {has_grads}")
        
        print(f"Model input cached: {model._input is not None}")
        print(f"Model output cached: {model._output is not None}")

    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_model(sys.argv[1])
    else:
        print("Usage: python inspect_model.py <model_file>")
