import sys
import os
import numpy as np
from nn.model import NeuralNetwork

def compress_model_object(model):
    print("Compressing model...")
    
    # 1. Convert weights and biases to float32
    print("- Converting weights to float32")
    for layer, _ in model._layers:
        if hasattr(layer, 'weights') and layer.weights is not None:
            layer.weights = layer.weights.astype(np.float32)
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias = layer.bias.astype(np.float32)
            
    # 2. Clear optimizer state
    print("- Clearing optimizer state")
    # Re-initialize optimizer to clear state (m_w, v_w, etc.)
    # We keep the configuration but lose the state
    opt_class = model._optimizer.__class__
    opt_vars = vars(model._optimizer)
    # Create a new fresh optimizer with same params
    # This is a bit hacky, better to just clear the dicts if we know them
    if hasattr(model._optimizer, 'm_w'): model._optimizer.m_w = {}
    if hasattr(model._optimizer, 'v_w'): model._optimizer.v_w = {}
    if hasattr(model._optimizer, 'm_b'): model._optimizer.m_b = {}
    if hasattr(model._optimizer, 'v_b'): model._optimizer.v_b = {}
    if hasattr(model._optimizer, 'cache_w'): model._optimizer.cache_w = {}
    if hasattr(model._optimizer, 'cache_b'): model._optimizer.cache_b = {}
    
    # 3. Clear gradients
    print("- Clearing gradients")
    for layer, _ in model._layers:
        layer.grad_weights = None
        layer.grad_bias = None
        layer._dw = None
        layer._db = None
        
    # 4. Clear cached inputs/outputs
    print("- Clearing cached inputs/outputs")
    model._input = None
    model._output = None
    model._num_examples = None
    for layer, _ in model._layers:
        layer._output = None
        layer._input_units = None # This might be needed for build, but usually build is done. 
                                  # Actually _input_units is just an int, keep it.
    return model

def compress_model(input_path, output_path):
    print(f"Loading model from {input_path}...")
    model = NeuralNetwork.load(input_path)
    
    compress_model_object(model)
    
    print(f"Saving compressed model to {output_path}...")
    model.save(output_path)
    
    original_size = os.path.getsize(input_path)
    new_size = os.path.getsize(output_path)
    
    print(f"Done!")
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"New size:      {new_size / 1024 / 1024:.2f} MB")
    print(f"Reduction:     {(1 - new_size/original_size)*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compress_nn.py <input_model> <output_model>")
    else:
        compress_model(sys.argv[1], sys.argv[2])
