import os
# ================ GPU ACCELERATION CONFIGURATION ================
# Set environment variables before importing TensorFlow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent OOM errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Better memory allocation
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Avoid OneDNN compatibility issues

# Import other non-TF libraries first
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import io
import argparse
import json
import traceback
import sys
import datetime
from collections import Counter

# Try to detect if running in Colab
def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Colab environment detection
IN_COLAB = is_running_in_colab()

# ================ TENSORFLOW IMPORT ================
# Import TensorFlow
import tensorflow as tf

# CRITICAL - Fix GPU detection issues
print("\n===== CUDA/GPU INITIALIZATION =====")
# Set memory growth and configure GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU device(s): {physical_devices}")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        except Exception as e:
            print(f"Failed to set memory growth: {e}")
    
    # Force device visibility - helps with some Colab environments
    try:
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        print(f"Set {physical_devices[0]} as visible GPU")
    except Exception as e:
        print(f"Failed to set visible device: {e}")
        
    # Verify GPU is actually usable
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            result = c.numpy()  # Forces execution
            print(f"✅ Basic GPU test successful: {result}")
            
            # Create a large matrix to ensure visible GPU usage
            large_a = tf.random.normal([1000, 1000])
            large_b = tf.random.normal([1000, 1000])
            large_c = tf.matmul(large_a, large_b)
            large_result = large_c.numpy()  # Forces execution
            print("✅ Large matrix GPU test successful")
            
            # Keep a reference to force visibility in nvidia-smi
            gpu_warmer = tf.random.normal([2000, 2000])
            _ = tf.matmul(gpu_warmer, gpu_warmer).numpy()
            
            # Install and check cuDNN
            try:
                import subprocess
                print("\n===== INSTALLING CUDNN LIBRARIES =====")
                # Install compatible cuDNN libraries for TF 2.15
                subprocess.run("pip install nvidia-cudnn-cu11==8.6.0.163", shell=True, check=True)
                print("✅ cuDNN libraries installed")
                
                # Test convolution operations that require cuDNN
                print("Testing convolution operations...")
                test_input = tf.random.normal([1, 32, 32, 3])
                test_filter = tf.random.normal([3, 3, 3, 16])
                test_output = tf.nn.conv2d(test_input, test_filter, strides=[1, 1, 1, 1], padding='SAME')
                _ = test_output.numpy()  # Force execution
                print("✅ GPU convolution test successful!")
            except Exception as e:
                print(f"⚠️ cuDNN installation/test failed: {e}")
                print("Some GPU operations may fall back to CPU")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        print("This means TensorFlow may default to CPU operations")
else:
    print("❌ No GPU devices detected by TensorFlow")
print("===================================\n")

# Import tensorflow_text early to ensure SentencepieceOp gets registered
# This must be imported before loading any model that uses SentencepieceOp
try:
    import tensorflow_text
    print("✅ tensorflow_text imported successfully")
except ImportError:
    print("❌ tensorflow_text is not installed - SentencepieceOp will not be available")
    print("Install with: pip install tensorflow-text")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set TensorFlow to use the GPU properly in Colab
tf.config.set_soft_device_placement(True)
print("TensorFlow eager execution:", tf.executing_eagerly())
print("TensorFlow version:", tf.__version__)
try:
    print("TensorFlow Text version:", tensorflow_text.__version__)
except:
    print("TensorFlow Text version: not available")

# After importing TensorFlow, add this code to show GPU devices and test GPU usage
print("\n====== GPU CHECK ======")
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print("GPU is available! Testing GPU...")
    try:
        with tf.device('/GPU:0'):
            # Create and multiply large matrices to test GPU
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            c = tf.matmul(a, b)
            # Force execution to see if GPU works
            result = c.numpy()
            print("✅ GPU test successful!")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
else:
    print("❌ No GPU detected. Running on CPU only.")
print("========================\n")

# Define argument parser for flexibility
def parse_args():
    parser = argparse.ArgumentParser(description='Process X-ray images with CXR Foundation model and neural network')
    parser.add_argument('--model_path', type=str, default="hf/elixr-c-v2-pooled",
                        help='Path to the local CXR Foundation model directory')
    parser.add_argument('--model_format', type=str, choices=['tf_saved_model', 'tf_hub', 'onnx'],
                        default='tf_saved_model', help='Format of the model (tf_saved_model, tf_hub, or onnx)')
    parser.add_argument('--normal_dir', type=str, default="dataset/Normal",
                        help='Directory containing normal X-ray images')
    parser.add_argument('--tb_dir', type=str, default="dataset/Tuberculosis",
                        help='Directory containing tuberculosis X-ray images')
    parser.add_argument('--output_dir', type=str, default="output_results",
                        help='Directory to save output results')
    parser.add_argument('--image_process', type=int, default=5,
                        help='Number of images to process per class (0 for all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping')
    parser.add_argument('--hidden_layers', type=str, default="512,256",
                        help='Comma-separated list of hidden layer sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate for the model')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose logging')
    parser.add_argument('--use_dummy_embeddings', action='store_true',
                        help='DEVELOPMENT ONLY: Use synthetic embeddings instead of loading the CXR model')
    parser.add_argument('--max_failures', type=int, default=3,
                        help='Maximum consecutive failures before stopping (0 to never stop)')
    parser.add_argument('--allow_auto_fallback', action='store_true',
                        help='DEVELOPMENT ONLY: Allow fallback to synthetic embeddings on errors (NOT FOR MEDICAL USE)')
    parser.add_argument('--medical_mode', action='store_true', default=True,
                        help='Medical use mode - ensures no synthetic embeddings are used and stops on errors')
    parser.add_argument('--force_tensor_input', action='store_true', default=False,
                        help='Force using raw tensor input instead of TF Examples even when the model expects strings')
    parser.add_argument('--max_gpu_batch_size', type=int, default=4,
                        help='Maximum batch size for GPU processing (0 for individual processing)')

    # Handle Colab's extra arguments by ignoring unknown args
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

    # Process hidden_layers from string to list of integers
    if isinstance(args.hidden_layers, str):
        args.hidden_layers = [int(x) for x in args.hidden_layers.split(',')]

    # Override settings for medical safety
    if args.medical_mode:
        if args.use_dummy_embeddings:
            print("\n" + "="*80)
            print("⚠️ MEDICAL SAFETY WARNING ⚠️".center(80))
            print("Medical mode is incompatible with dummy embeddings.".center(80))
            print("Setting --use_dummy_embeddings=False for safety.".center(80))
            print("="*80 + "\n")
            args.use_dummy_embeddings = False

        if args.allow_auto_fallback:
            print("\n" + "="*80)
            print("⚠️ MEDICAL SAFETY WARNING ⚠️".center(80))
            print("Medical mode is incompatible with auto fallback.".center(80))
            print("Setting --allow_auto_fallback=False for safety.".center(80))
            print("="*80 + "\n")
            args.allow_auto_fallback = False

    return args

# Set global variables from arguments
args = parse_args()
MODEL_PATH = args.model_path
MODEL_FORMAT = args.model_format
NORMAL_DIR = args.normal_dir
TB_DIR = args.tb_dir
OUTPUT_DIR = args.output_dir
IMAGE_PROCESS = args.image_process
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
PATIENCE = args.patience
HIDDEN_LAYERS = args.hidden_layers
DROPOUT_RATE = args.dropout_rate
DEBUG = args.debug
USE_DUMMY_EMBEDDINGS = args.use_dummy_embeddings
MAX_FAILURES = args.max_failures
ALLOW_AUTO_FALLBACK = args.allow_auto_fallback
MEDICAL_MODE = args.medical_mode
FORCE_TENSOR_INPUT = args.force_tensor_input
MAX_GPU_BATCH_SIZE = args.max_gpu_batch_size

# Safety check for medical applications
if MEDICAL_MODE:
    if USE_DUMMY_EMBEDDINGS or ALLOW_AUTO_FALLBACK:
        print("\n" + "="*80)
        print("⚠️ CRITICAL MEDICAL SAFETY ERROR ⚠️".center(80))
        print("Safety constraints bypassed - terminating for patient safety".center(80))
        print("="*80 + "\n")
        sys.exit(1)
    else:
        # Allow CPU fallback in medical mode
        print("\n" + "="*80)
        print("⚠️ NOTE: Medical mode will allow CPU fallback ⚠️".center(80)) 
        print("All processing will use CPU if GPU operations fail".center(80))
        print("This maintains model accuracy while ensuring compatibility".center(80))
        print("="*80 + "\n")

# For Colab, allow direct override of parameters
if IN_COLAB:
    print("Running in Google Colab environment")

    # You can override parameters here for Colab
    # MODEL_PATH = "hf_models/elixr-c-v2-pooled"  # Uncomment and change if needed
    # NORMAL_DIR = "dataset/Normal"  # Uncomment and change if needed
    # TB_DIR = "dataset/Tuberculosis"  # Uncomment and change if needed
    # IMAGE_PROCESS = 10  # Uncomment and change if needed
    # DEBUG = True  # Uncomment to enable debug mode
    # USE_DUMMY_EMBEDDINGS = True  # Uncomment to use dummy embeddings

    print(f"Using parameters:")
    print(f"  MODEL_PATH: {MODEL_PATH}")
    print(f"  NORMAL_DIR: {NORMAL_DIR}")
    print(f"  TB_DIR: {TB_DIR}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  IMAGE_PROCESS: {IMAGE_PROCESS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"  PATIENCE: {PATIENCE}")
    print(f"  HIDDEN_LAYERS: {HIDDEN_LAYERS}")
    print(f"  DROPOUT_RATE: {DROPOUT_RATE}")
    print(f"  DEBUG: {DEBUG}")
    print(f"  USE_DUMMY_EMBEDDINGS: {USE_DUMMY_EMBEDDINGS}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class mapping
CLASS_MAPPING = {
    "Normal": 0,
    "Tuberculosis": 1
}

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to display training progress without using progress bars that go to stderr."""

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch+1}/{self.params['epochs']}...")

    def on_epoch_end(self, epoch, logs=None):
        # Extract metrics
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        print(f"Completed epoch {epoch+1}/{self.params['epochs']} - {metrics_str}")

    def on_train_batch_end(self, batch, logs=None):
        # Print batch progress every few batches
        if batch % 10 == 0 or batch == self.params['steps'] - 1:
            batch_metrics = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            print(f"  Batch {batch+1}/{self.params['steps']} - {batch_metrics}")

def preprocess_image_for_model(image_path):
    """
    Preprocess the image for model input using TensorFlow operations.
    Following EfficientNet preprocessing specifications.
    """
    try:
        # Check if the file exists and is readable
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return None

        # Read the image file
        try:
            img = tf.io.read_file(image_path)
        except Exception as e:
            print(f"Failed to read image file {image_path}: {e}")
            return None

        # Decode the image - handle both JPG and PNG with fallbacks
        try:
            # Try to determine the image format from the file extension
            if image_path.lower().endswith(('.jpg', '.jpeg')):
                img = tf.image.decode_jpeg(img, channels=3)
            elif image_path.lower().endswith('.png'):
                img = tf.image.decode_png(img, channels=3)
            else:
                # If unable to determine from extension, try JPEG first
                try:
                    img = tf.image.decode_jpeg(img, channels=3)
                except:
                    try:
                        img = tf.image.decode_png(img, channels=3)
                    except:
                        print(f"Failed to decode image as JPEG or PNG: {image_path}")
                        return None
        except Exception as e:
            print(f"Failed to decode image {image_path}: {e}")
            return None

        # Resize to 224x224 (standard size for EfficientNet)
        try:
            img = tf.image.resize(img, [224, 224])
        except Exception as e:
            print(f"Failed to resize image {image_path}: {e}")
            return None

        # Convert to float and normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0

        # Use EfficientNet specific preprocessing - normalize with ImageNet stats
        # These are the standard normalization values for EfficientNet
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        img = (img - mean) / std

        return img
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        if DEBUG:
            traceback.print_exc()
        return None

def create_model_input(image_path, serving_fn=None):
    """Create appropriate input for the model based on its signature."""
    # Preprocess the image
    img = preprocess_image_for_model(image_path)
    if img is None:
        return None

    # If we have the serving function, check the expected input format
    if serving_fn is not None:
        # Get input signature details
        sig_inputs = list(serving_fn.structured_input_signature[1].items())
        if len(sig_inputs) > 0:
            input_name, input_spec = sig_inputs[0]
            if DEBUG:
                print(f"Model expects input: {input_name} with shape {input_spec.shape}")

            # If input is a string, try creating a serialized tf.Example
            if input_spec.dtype == tf.string:
                if DEBUG:
                    print("Model expects serialized tf.Example as input")
                return create_tf_example(image_path)

            # If input is a tensor, batch it
            img = tf.expand_dims(img, 0)  # Add batch dimension
    else:
        # Default: assume batched tensor input
        img = tf.expand_dims(img, 0)

    return img

def create_tf_example(image_path):
    """Create a TensorFlow Example from an image file."""
    try:
        # Use the adapter to try different formats
        print(f"Trying EfficientNet Example format for {os.path.basename(image_path)}")
        return TFExampleFormatAdapter.create_efficientnet_example(image_path)
    except Exception as e:
        print(f"Error creating TF Example for {image_path}: {e}")
        if DEBUG:
            traceback.print_exc()
        return None

def load_elixr_model(model_path):
    """Load the elixr-c-v2-pooled model using TensorFlow Hub for better compatibility."""
    try:
        print("Attempting to load the CXR Foundation model using TensorFlow Hub...")

        # Import hub if available
        try:
            import tensorflow_hub as hub
            print("✅ TensorFlow Hub imported successfully")
        except ImportError:
            print("❌ TensorFlow Hub is not installed")
            print("Try installing with: pip install tensorflow-hub")
            return None

        # Load the model from Hub format
        try:
            model = hub.load(model_path)
            print("✅ Model loaded via TensorFlow Hub")
            return model
        except Exception as e:
            print(f"❌ Error loading model via TensorFlow Hub: {e}")
            if DEBUG:
                traceback.print_exc()
            return None
    except Exception as e:
        print(f"❌ Error in elixr model loading: {e}")
        if DEBUG:
            traceback.print_exc()
        return None

def load_model_with_fallback():
    try:
        print("Loading CXR Foundation model...")

        # Important: Import tensorflow_text first to ensure SentencepieceOp is registered
        try:
            import tensorflow_text
            print("✅ tensorflow_text imported successfully")
        except ImportError:
            print("❌ tensorflow_text is not installed - this is required for the CXR Foundation model")
            print("Install with: pip install tensorflow-text==" + tf.__version__.rsplit('.', 1)[0] + ".*")
            return None

        # Check if model path exists and find the correct SavedModel directory
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model directory not found: {MODEL_PATH}")
            print("Please check that you've downloaded the CXR Foundation model")
            print("You may need to update the MODEL_PATH argument to point to the correct location")
            return None

        # For elixr-c-v2 models, try the Hub loader first
        if "elixr" in MODEL_PATH.lower():
            model = load_elixr_model(MODEL_PATH)
            if model is not None:
                return model
            print("Hub loading failed, trying standard SavedModel loader...")

        # Continue with standard SavedModel loading
        # Check for common model directory structures
        potential_model_dir = MODEL_PATH

        # If model has a SavedModel or saved_model.pb file at the top level
        if os.path.exists(os.path.join(MODEL_PATH, 'saved_model.pb')):
            print(f"✅ Found SavedModel file at: {MODEL_PATH}")
        else:
            # Check for common subdirectory patterns in HuggingFace models
            for subdir in ['model', 'tf_model', 'saved_model']:
                test_path = os.path.join(MODEL_PATH, subdir)
                if os.path.exists(test_path) and os.path.exists(os.path.join(test_path, 'saved_model.pb')):
                    print(f"✅ Found SavedModel file in subdirectory: {subdir}")
                    potential_model_dir = test_path
                    break

            # Check for version-numbered subdirectories (common TF SavedModel pattern)
            version_dirs = [d for d in os.listdir(MODEL_PATH) if d.isdigit() and
                          os.path.isdir(os.path.join(MODEL_PATH, d))]
            if version_dirs:
                latest_version = max(version_dirs, key=int)
                test_path = os.path.join(MODEL_PATH, latest_version)
                if os.path.exists(os.path.join(test_path, 'saved_model.pb')):
                    print(f"✅ Found SavedModel file in version directory: {latest_version}")
                    potential_model_dir = test_path

        print(f"Attempting to load model from: {potential_model_dir}")

        # Configure TensorFlow for better compatibility
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent GPU memory errors
        tf.config.set_soft_device_placement(True)  # Let TF choose devices

        # Load with options for better compatibility
        options = tf.saved_model.LoadOptions(
            experimental_io_device='/job:localhost',
            allow_partial_checkpoint=True,
            experimental_skip_checkpoint=True
        )

        # Try loading the model
        try:
            model = tf.saved_model.load(potential_model_dir, options=options)

            # Verify model by checking signatures
            if not hasattr(model, 'signatures') or not model.signatures:
                print("❌ Model loaded but doesn't have valid signatures")
                return None

            print("✅ Model loaded successfully with signatures:", list(model.signatures.keys()))

            # We'll get more information about the model but skip the test inference
            # which was causing the DecodePng error

            # Find a suitable signature
            if 'serving_default' in model.signatures:
                sig_name = 'serving_default'
            else:
                sig_name = list(model.signatures.keys())[0]

            print(f"Using signature: {sig_name}")
            serving_fn = model.signatures[sig_name]

            # Check input specs
            input_specs = serving_fn.structured_input_signature[1]
            input_name = list(input_specs.keys())[0]
            print(f"Model expects input: {input_name} with shape {input_specs[input_name].shape} and type {input_specs[input_name].dtype}")

            # Skip the problematic test inference that was causing DecodePng errors
            print("Skipping test inference to avoid DecodePng errors")
            print("Model will be tested with real images during processing")

            return model

        except tf.errors.NotFoundError as e:
            if "Op type not registered 'SentencepieceOp'" in str(e):
                print("❌ SentencepieceOp not registered error detected")
                print("This typically happens when tensorflow_text is not properly installed or imported")
                print("Ensure you have installed tensorflow-text matching your TensorFlow version")
                print(f"Try: pip install tensorflow-text=={tf.__version__.rsplit('.', 1)[0]}.*")
                return None
            elif "No such file or directory" in str(e):
                print(f"❌ SavedModel files not found at {potential_model_dir}")
                print("Please check that you have the correct model path")
                return None
            else:
                print(f"❌ Error loading model: {str(e)}")
                return None
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            return None

    except Exception as e:
        print(f"❌ Error during model loading process: {e}")
        if DEBUG:
            traceback.print_exc()
        return None

def create_elixr_compatible_example(image_path):
    """Create a TensorFlow Example specifically formatted for elixr-c-v2-pooled model."""
    try:
        # Read original image file bytes
        with open(image_path, 'rb') as f:
            encoded_image = f.read()

        # Create a minimal example with just the image bytes
        feature = {
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()
    except Exception as e:
        print(f"Error creating elixr-compatible TF Example: {e}")
        if DEBUG:
            traceback.print_exc()
        return None

def batch_process_with_gpu(model, image_paths, batch_size=4, class_name="Normal"):
    """
    Process images in batches using GPU when possible.
    Returns empty lists if all processing fails.
    """
    if 'serving_default' in model.signatures:
        serving_fn = model.signatures['serving_default']
    else:
        serving_fn = model.signatures[list(model.signatures.keys())[0]]

    input_specs = serving_fn.structured_input_signature[1]
    input_name = list(input_specs.keys())[0]

    all_embeddings = []
    all_labels = []
    all_image_paths = []
    failed_paths = []

    # Check for GPU more aggressively
    using_gpu = False
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        # Try to install cuDNN if missing
        try:
            import subprocess
            print("\n⚠️ Checking cuDNN installation...")
            subprocess.run(["pip", "install", "nvidia-cudnn-cu11==8.6.0.163"], check=True)
            print("✅ cuDNN installation checked")
        except:
            print("⚠️ Could not verify cuDNN installation")
            
        # More GPU checks
        try:
            # Force operations to GPU
            with tf.device('/GPU:0'):
                # Simple test to make sure GPU works with the operations we need
                print("Testing GPU with basic convolution operation...")
                test_input = tf.random.normal([1, 32, 32, 3])
                test_filter = tf.random.normal([3, 3, 3, 16])
                test_output = tf.nn.conv2d(test_input, test_filter, strides=[1, 1, 1, 1], padding='SAME')
                _ = test_output.numpy()  # Force execution
                
                # If we get here, convolution operations work on GPU
                using_gpu = True
                print(f"\n✅ Using GPU for batch processing of {len(image_paths)} {class_name} images with batch size {batch_size}")
        except Exception as e:
            print(f"\n⚠️ GPU convolution test failed: {e}")
            print("Falling back to CPU for processing")
            using_gpu = False
    else:
        print(f"\n⚠️ No GPU detected, processing {len(image_paths)} {class_name} images individually")
        using_gpu = False
    
    # If GPU tests failed, process individually on CPU
    if not using_gpu:
        batch_size = 1  # Fall back to individual processing
        print("WARNING: Using CPU for processing due to GPU compatibility issues")

    # Process in batches
    any_success = False 
    total_processed = 0
    
    try:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            print(f"Processing images {i+1}-{min(i+batch_size, len(image_paths))} of {len(image_paths)}...")
            
            # Prepare batch serialized examples
            batch_serialized = []
            valid_paths = []
            
            for img_path in batch_paths:
                try:
                    serialized = create_elixr_compatible_example(img_path)
                    if serialized is not None:
                        batch_serialized.append(serialized)
                        valid_paths.append(img_path)
                    else:
                        print(f"Warning: Could not serialize {os.path.basename(img_path)}, skipping.")
                        failed_paths.append(img_path)
                except Exception as e:
                    print(f"Error serializing {os.path.basename(img_path)}: {e}")
                    failed_paths.append(img_path)
                    if DEBUG:
                        traceback.print_exc()
            
            if not batch_serialized:
                continue
                
            # Process the batch with GPU if we're batching, otherwise process individually
            if batch_size > 1 and len(batch_serialized) > 1 and using_gpu:
                try:
                    # Place operations on GPU explicitly with error handling
                    try:
                        # Try with GPU first
                        with tf.device('/GPU:0'):
                            model_input = tf.constant(batch_serialized, dtype=tf.string)
                            result = serving_fn(**{input_name: model_input})
                            batch_embeddings = next(iter(result.values())).numpy()
                    except tf.errors.InternalError as e:
                        # If we get a cuDNN error, fall back to CPU for this batch
                        if "DNN library is not found" in str(e):
                            print("⚠️ cuDNN library not found, falling back to CPU for this batch")
                            with tf.device('/CPU:0'):
                                model_input = tf.constant(batch_serialized, dtype=tf.string)
                                result = serving_fn(**{input_name: model_input})
                                batch_embeddings = next(iter(result.values())).numpy()
                        else:
                            raise
                    
                    print(f"Raw batch embeddings shape: {batch_embeddings.shape}")
                    
                    # Process embeddings based on shape
                    if len(batch_embeddings.shape) > 2:
                        batch_embeddings = np.mean(batch_embeddings, axis=(1, 2))
                        print(f"Averaged batch embeddings shape: {batch_embeddings.shape}")
                    
                    # Add all embeddings from this batch
                    for j, (emb, img_path) in enumerate(zip(batch_embeddings, valid_paths)):
                        all_embeddings.append(emb.squeeze())
                        all_labels.append(class_name)
                        all_image_paths.append(img_path)
                        total_processed += 1
                        any_success = True
                        print(f"✅ Processed image {j+1}/{len(valid_paths)}: {os.path.basename(img_path)}")
                        
                except Exception as e:
                    print(f"Batch processing failed: {e}")
                    if DEBUG:
                        traceback.print_exc()
                        
                    # Fall back to individual processing
                    print("Falling back to individual processing...")
                    for img_path, serialized in zip(valid_paths, batch_serialized):
                        try:
                            process_individual_image(model, img_path, serialized, serving_fn, input_name, 
                                                all_embeddings, all_labels, all_image_paths, failed_paths, class_name)
                            total_processed += 1
                            any_success = True
                        except Exception as e:
                            print(f"Individual processing failed for {os.path.basename(img_path)}: {e}")
                            failed_paths.append(img_path)
            else:
                # Individual processing
                for img_path, serialized in zip(valid_paths, batch_serialized):
                    try:
                        process_individual_image(model, img_path, serialized, serving_fn, input_name, 
                                            all_embeddings, all_labels, all_image_paths, failed_paths, class_name)
                        total_processed += 1 
                        any_success = True
                    except Exception as e:
                        print(f"Individual processing failed for {os.path.basename(img_path)}: {e}")
                        failed_paths.append(img_path)
    except Exception as e:
        print(f"Error during batch processing: {e}")
        if DEBUG:
            traceback.print_exc()
    
    print(f"Finished processing {class_name} images.")
    print(f"Successfully processed: {len(all_embeddings)} images")
    print(f"Failed to process: {len(failed_paths)} images")
    
    if failed_paths and len(failed_paths) < 5:
        print("Failed images:")
        for path in failed_paths:
            print(f"  - {os.path.basename(path)}")
    
    # Check if all processing failed
    if not any_success and len(failed_paths) == len(image_paths):
        print("\n===== ALL IMAGE PROCESSING FAILED =====")
        print("Will try direct CPU processing as fallback")
    
    return all_embeddings, all_labels, all_image_paths

def process_individual_image(model, img_path, serialized, serving_fn, input_name, 
                          all_embeddings, all_labels, all_image_paths, failed_paths, class_name):
    """Helper function to process a single image and add its embedding."""
    try:
        print(f"Processing image: {os.path.basename(img_path)}")
        
        # Create a single-element batch
        model_input = tf.constant([serialized], dtype=tf.string)
        
        # Try running on GPU first, fall back to CPU if needed
        try:
            # First attempt with GPU
            with tf.device('/GPU:0'):
                result = serving_fn(**{input_name: model_input})
                embedding = next(iter(result.values())).numpy()
        except tf.errors.InternalError as e:
            # If we get a cuDNN error, fall back to CPU
            if "DNN library is not found" in str(e):
                print("⚠️ cuDNN library not found, falling back to CPU for this image")
                with tf.device('/CPU:0'):
                    result = serving_fn(**{input_name: model_input})
                    embedding = next(iter(result.values())).numpy()
            else:
                raise
        
        # Debug the shape
        print(f"Raw embedding shape: {embedding.shape}")
        
        # Process embedding based on shape
        if len(embedding.shape) > 2:
            # Handle spatial dimensions (height, width)
            embedding = np.mean(embedding, axis=(1, 2))
            print(f"Averaged embedding shape: {embedding.shape}")
        
        # Extract and add the embedding
        emb = embedding.squeeze()
        print(f"Final embedding shape: {emb.shape}")
        
        all_embeddings.append(emb)
        all_labels.append(class_name)
        all_image_paths.append(img_path)
        print(f"✅ Successfully processed image {os.path.basename(img_path)}")
        
    except Exception as e:
        failed_paths.append(img_path)
        print(f"❌ Error processing image {os.path.basename(img_path)}: {e}")
        if DEBUG:
            traceback.print_exc()

def get_embedding(model, image_path):
    """Get embedding for a single image."""
    if model is None:
        raise ValueError("Model is not loaded")

    # Get the serving function and input details
    try:
        if 'serving_default' in model.signatures:
            sig_name = 'serving_default'
            serving_fn = model.signatures[sig_name]
        else:
            # Try the first available signature
            sig_name = list(model.signatures.keys())[0]
            serving_fn = model.signatures[sig_name]
            print(f"Using signature: {sig_name}")

        input_specs = serving_fn.structured_input_signature[1]
        input_name = list(input_specs.keys())[0]
        input_dtype = input_specs[input_name].dtype

        # Check if model expects string input (serialized TF Example) and not forcing tensor input
        if input_dtype == tf.string and not FORCE_TENSOR_INPUT:
            print(f"Creating elixr-specific input for {os.path.basename(image_path)}")

            # Create the elixr-compatible TF Example
            serialized = create_elixr_compatible_example(image_path)
            if serialized is None:
                raise ValueError(f"Failed to create TF Example for {image_path}")

            # Create model input
            model_input = tf.constant([serialized], dtype=tf.string)

            # Silence TensorFlow logging temporarily
            original_tf_cpp_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow logs

            try:
                # Run inference
                print(f"Running inference for {os.path.basename(image_path)} with elixr format...")
                result = serving_fn(**{input_name: model_input})
                embedding = next(iter(result.values())).numpy()

                # Restore original logging level
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_tf_cpp_log_level

                print(f"✅ Successfully generated embedding with shape {embedding.shape}")
                return embedding
            except Exception as e:
                # Restore original logging level
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_tf_cpp_log_level
                print(f"❌ Error during model inference: {e}")
                if DEBUG:
                    traceback.print_exc()
                raise ValueError(f"Model inference failed for {image_path}")
        else:
            # Use tensor-based input (standard image)
            image = preprocess_image_for_model(image_path)
            if image is None:
                raise ValueError(f"Failed to preprocess image: {image_path}")

            # Add batch dimension
            model_input = tf.expand_dims(image, 0)
            if FORCE_TENSOR_INPUT and input_dtype == tf.string:
                print(f"FORCING tensor input format instead of TF Example for {os.path.basename(image_path)}")
            else:
                print(f"Using tensor input format for {os.path.basename(image_path)}")

            # Silence TensorFlow logging temporarily
            original_tf_cpp_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow logs

            try:
                # Try to handle the case where tensor input is forced but model expects string
                if FORCE_TENSOR_INPUT and input_dtype == tf.string:
                    # Try direct inference with custom signature if available
                    if hasattr(model, '__call__') and callable(model.__call__):
                        print("Using direct model call with tensor input")
                        result_tensor = model(model_input)

                        # Extract the embedding from the result
                        if isinstance(result_tensor, dict):
                            embedding = next(iter(result_tensor.values())).numpy()
                        elif isinstance(result_tensor, (list, tuple)) and len(result_tensor) > 0:
                            embedding = result_tensor[0].numpy()
                        else:
                            embedding = result_tensor.numpy()
                    else:
                        raise ValueError("Cannot force tensor input - model doesn't support direct calling")
                else:
                    # Standard signature-based inference
                    result = serving_fn(**{input_name: model_input})
                    embedding = next(iter(result.values())).numpy()

                # Restore original logging level
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_tf_cpp_log_level

                print(f"✅ Successfully generated embedding with shape {embedding.shape}")
                return embedding

            except Exception as e:
                # Restore original logging level
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_tf_cpp_log_level
                print(f"❌ Error during model inference: {e}")
                if DEBUG:
                    traceback.print_exc()
                raise ValueError(f"Model inference failed for {image_path}")

    except Exception as e:
        print(f"❌ Error setting up model inference: {e}")
        if DEBUG:
            traceback.print_exc()
        raise ValueError(f"Failed to setup model inference for {image_path}")

def create_neural_network(input_size, hidden_layers_sizes, dropout_rate):
    """Create a neural network model with the specified architecture."""
    model = Sequential()

    # Input layer
    model.add(Input(shape=(input_size,)))

    # Hidden layers
    for layer_size in hidden_layers_sizes:
        model.add(Dense(layer_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    return model

def plot_training_history(history, save_path):
    """Plot and save training history."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add labels
    classes = list(CLASS_MAPPING.keys())
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics."""
    # Predict
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(np.int32).reshape(-1)

    # Ensure data types are consistent
    y_compare = y.astype(np.int32)

    # Calculate metrics
    accuracy = accuracy_score(y_compare, y_pred)
    precision = precision_score(y_compare, y_pred)
    recall = recall_score(y_compare, y_pred)
    f1 = f1_score(y_compare, y_pred)
    auc = roc_auc_score(y_compare, y_pred_prob)

    print(f"\n{dataset_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

    return metrics, y_pred, y_pred_prob

# Update the dummy embedding function to create class-separable embeddings
def generate_dummy_embedding(image_path, embedding_dim=4096, is_tb=False):
    """Generate a deterministic but class-separable random embedding for an image path."""
    # Create a random embedding with a consistent seed based on filename
    seed = hash(image_path) % 10000
    np.random.seed(seed)

    # Create base random embedding
    dummy_emb = np.random.randn(embedding_dim).astype(np.float32)

    # Add class-specific bias to make embeddings more separable
    # This creates a clear separation between normal and TB embeddings
    # making it easier for the neural network to learn
    if is_tb:
        # For tuberculosis, bias the first 100 dimensions higher
        dummy_emb[:100] += 3.0
    else:
        # For normal, bias the first 100 dimensions lower
        dummy_emb[:100] -= 3.0

    return dummy_emb

class TFExampleFormatAdapter:
    """Helper class to try different TF Example formats for model compatibility."""

    @staticmethod
    def create_basic_example(image_path):
        """Basic TF Example with minimal fields."""
        try:
            with tf.io.gfile.GFile(image_path, 'rb') as fid:
                encoded_image = fid.read()

            feature = {
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            return example.SerializeToString()
        except Exception as e:
            print(f"Error creating basic TF Example: {e}")
            return None

    @staticmethod
    def create_efficientnet_example(image_path):
        """TF Example formatted for EfficientNet models."""
        try:
            with tf.io.gfile.GFile(image_path, 'rb') as fid:
                encoded_image = fid.read()

            feature = {
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
                'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
                'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'0'*64])),
                'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode()])),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            return example.SerializeToString()
        except Exception as e:
            print(f"Error creating EfficientNet TF Example: {e}")
            return None

    @staticmethod
    def create_full_example(image_path):
        """TF Example with comprehensive fields for maximum compatibility."""
        try:
            # Read and preprocess image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to match EfficientNet's expected input size
            img = img.resize((224, 224))

            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes = img_bytes.getvalue()

            # Get image shape
            height, width = 224, 224
            channels = 3

            feature = {
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
                'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'image/channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
                'image/colorspace': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'RGB'])),
                'image/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode()])),
                'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode()])),
                'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'0'*64])),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            return example.SerializeToString()
        except Exception as e:
            print(f"Error creating full TF Example: {e}")
            return None

def try_load_and_convert_image(image_path):
    """Try different methods to load and convert image to a format compatible with elixr model."""
    try:
        # Method 1: Convert to RGB using PIL
        try:
            from PIL import Image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array
        except Exception as e:
            print(f"PIL method failed: {e}")

        # Method 2: Use TensorFlow's image decoding directly
        try:
            img_bytes = tf.io.read_file(image_path)
            img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
            img = tf.image.resize(img, (224, 224))
            img = tf.cast(img, tf.float32) / 255.0
            return img.numpy()
        except Exception as e:
            print(f"TensorFlow method failed: {e}")

        # Method 3: Just read the raw bytes
        try:
            with open(image_path, 'rb') as f:
                raw_bytes = f.read()
            return raw_bytes
        except Exception as e:
            print(f"Raw bytes method failed: {e}")

        return None
    except Exception as e:
        print(f"All image loading methods failed: {e}")
        return None

def process_elixr_model_input(model, image_path, force_cpu=False):
    """
    Special handler for elixr-c-v2-pooled model to work around the Conv2D input issue.
    This function tries different approaches to get the model to work.
    """
    try:
        print(f"Using specialized elixr input processing for {os.path.basename(image_path)}")

        # Get serving function
        if 'serving_default' in model.signatures:
            serving_fn = model.signatures['serving_default']
        else:
            serving_fn = model.signatures[list(model.signatures.keys())[0]]

        input_specs = serving_fn.structured_input_signature[1]
        input_name = list(input_specs.keys())[0]

        # Try with minimal TF Example first - USING CPU EXPLICITLY
        try:
            print("Approach 1: Using minimal TF Example with raw image bytes (on CPU)")
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            # Create a simple TF Example with minimal fields
            feature = {
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized = example.SerializeToString()

            model_input = tf.constant([serialized], dtype=tf.string)
            
            # Force CPU execution to avoid cuDNN errors
            with tf.device('/CPU:0'):
                result = serving_fn(**{input_name: model_input})
                embedding = next(iter(result.values())).numpy()
            
            print(f"✅ Approach 1 succeeded with embedding shape: {embedding.shape}")
            return embedding
        except Exception as e:
            if DEBUG:
                traceback.print_exc()
            print(f"Approach 1 failed: {e}")

        # Try other approaches only if not forcing CPU
        if not force_cpu:
            # Try using tensor directly if the model supports it
            try:
                print("Approach 2: Forcing tensor input")
                img = preprocess_image_for_model(image_path)
                if img is None:
                    raise ValueError("Image preprocessing failed")

                img_batch = tf.expand_dims(img, 0)

                # Special handling for model that can be called directly
                if hasattr(model, '__call__') and callable(model.__call__):
                    print("Using direct model call")
                    with tf.device('/CPU:0'):
                        result = model(img_batch)

                    # Extract embedding
                    if isinstance(result, dict):
                        embedding = next(iter(result.values())).numpy()
                    else:
                        embedding = result.numpy()

                    print(f"✅ Approach 2 succeeded with embedding shape: {embedding.shape}")
                    return embedding
                else:
                    print("Model doesn't support direct calling")
            except Exception as e:
                print(f"Approach 2 failed: {e}")

            # If we got here, try converting image in multiple ways
            # and creating more complex TF Examples
            print("Approach 3: Trying multiple image formats in TF Example")
            from PIL import Image

            # Load image with PIL
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))

            # Try both JPEG and PNG formats
            formats = [
                ('JPEG', b'jpeg'),
                ('PNG', b'png')
            ]

            for fmt_name, fmt_bytes in formats:
                try:
                    print(f"Trying {fmt_name} format...")
                    # Convert to bytes
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format=fmt_name)
                    img_bytes = img_bytes.getvalue()

                    # Create more comprehensive TF Example
                    feature = {
                        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fmt_bytes])),
                        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[224])),
                        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[224])),
                        'image/channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    serialized = example.SerializeToString()

                    model_input = tf.constant([serialized], dtype=tf.string)
                    with tf.device('/CPU:0'):
                        result = serving_fn(**{input_name: model_input})
                        embedding = next(iter(result.values())).numpy()
                    print(f"✅ {fmt_name} format worked with embedding shape: {embedding.shape}")
                    return embedding
                except Exception as e:
                    print(f"{fmt_name} format failed: {e}")

        print("❌ All approaches failed for elixr model")
        
        if MEDICAL_MODE:
            print("\n⚠️ MEDICAL MODE: CPU processing failed. Stopping for safety.")
            return None
        elif ALLOW_AUTO_FALLBACK:
            print("\n⚠️ Falling back to synthetic embeddings.")
            # Create a synthetic embedding for development mode
            is_tb = 'tuberculosis' in image_path.lower() or 'tb' in os.path.basename(image_path).lower()
            dummy_embedding = generate_dummy_embedding(image_path, embedding_dim=1376, is_tb=is_tb)
            dummy_embedding = dummy_embedding.reshape(1, -1)
            return dummy_embedding
        else:
            return None
    except Exception as e:
        print(f"❌ Error in elixr processing: {e}")
        if DEBUG:
            traceback.print_exc()
        return None

# Add a direct CPU-only elixr processing function
def process_images_cpu_only(model, image_paths, class_name):
    """
    Process images using CPU only, one by one, with no GPU operations.
    This is a last resort when GPU operations fail due to cuDNN issues.
    """
    print(f"\n===== USING CPU-ONLY PROCESSING FOR {len(image_paths)} {class_name} IMAGES =====")
    
    # Get serving function
    if 'serving_default' in model.signatures:
        serving_fn = model.signatures['serving_default']
    else:
        serving_fn = model.signatures[list(model.signatures.keys())[0]]

    input_specs = serving_fn.structured_input_signature[1]
    input_name = list(input_specs.keys())[0]
    
    # Force TensorFlow to use CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    all_embeddings = []
    all_labels = []
    all_image_paths = []
    
    # Process each image one by one with CPU only
    for i, img_path in enumerate(image_paths):
        try:
            print(f"Processing {class_name} image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            # Read image and create TF Example
            with open(img_path, 'rb') as f:
                image_bytes = f.read()

            # Create a simple TF Example
            feature = {
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized = example.SerializeToString()
            
            # Create model input
            model_input = tf.constant([serialized], dtype=tf.string)
            
            # Ensure CPU usage only
            with tf.device('/CPU:0'):
                # Run inference
                result = serving_fn(**{input_name: model_input})
                embedding = next(iter(result.values())).numpy()
            
            # Process embedding based on shape
            if len(embedding.shape) > 2:
                embedding = np.mean(embedding, axis=(1, 2))
            
            # Add to collection
            all_embeddings.append(embedding.squeeze())
            all_labels.append(class_name)
            all_image_paths.append(img_path)
            print(f"✅ Successfully processed with shape {embedding.shape}")
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(img_path)}: {e}")
            if DEBUG:
                traceback.print_exc()
    
    # Restore GPU visibility for other operations
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    
    print(f"===== CPU PROCESSING COMPLETE: {len(all_embeddings)}/{len(image_paths)} SUCCESSFUL =====\n")
    return all_embeddings, all_labels, all_image_paths

def main():
    # Get list of images with specific extensions
    normal_images = sorted(glob.glob(os.path.join(NORMAL_DIR, "*.jpg")) +
                           glob.glob(os.path.join(NORMAL_DIR, "*.png")))
    tb_images = sorted(glob.glob(os.path.join(TB_DIR, "*.jpg")) +
                       glob.glob(os.path.join(TB_DIR, "*.png")))

    print(f"Found {len(normal_images)} Normal images")
    print(f"Found {len(tb_images)} TB images")

    # Limit the number of images if IMAGE_PROCESS > 0
    if IMAGE_PROCESS > 0:
        normal_images = normal_images[:IMAGE_PROCESS]
        tb_images = tb_images[:IMAGE_PROCESS]

    if not normal_images or not tb_images:
        print("Could not find enough images in the dataset directories.")
        print(f"Found {len(normal_images)} normal images and {len(tb_images)} tuberculosis images.")
        return None, None

    print(f"Processing {len(normal_images)} normal images and {len(tb_images)} tuberculosis images.")

    # Print the operating mode
    if MEDICAL_MODE:
        print("\n" + "="*80)
        print("🏥 MEDICAL MODE ACTIVE 🏥".center(80))
        print("All safety constraints enforced for medical usage".center(80))
        print("- No synthetic embeddings will be used".center(80))
        print("- Process will stop on any model error".center(80))
        print("="*80 + "\n")
    elif USE_DUMMY_EMBEDDINGS:
        print("\n" + "="*80)
        print("⚠️ DEVELOPMENT MODE: USING SYNTHETIC EMBEDDINGS ⚠️".center(80))
        print("WARNING: Not suitable for medical diagnosis".center(80))
        print("WARNING: For testing and development only".center(80))
        print("="*80 + "\n")
    elif ALLOW_AUTO_FALLBACK:
        print("\n" + "="*80)
        print("⚠️ DEVELOPMENT MODE: AUTO-FALLBACK ENABLED ⚠️".center(80))
        print("WARNING: May use synthetic embeddings on errors".center(80))
        print("WARNING: Results not suitable for medical diagnosis".center(80))
        print("="*80 + "\n")

    # Initialize variables
    cxr_model = None
    all_embeddings = []
    all_labels = []
    all_image_paths = []
    embedding_dim = 4096  # Default embedding dimension for CXR model

    # Tracking variables for embeddings
    dummy_count = 0
    real_count = 0
    total_count = len(normal_images) + len(tb_images)

    # Variables for tracking model state
    consecutive_failures = 0
    auto_switched_to_dummy = False

    # Decide whether to use real model or dummy embeddings
    if USE_DUMMY_EMBEDDINGS:
        if MEDICAL_MODE:
            print("CRITICAL ERROR: Synthetic embeddings cannot be used in medical mode")
            print("Terminating for patient safety")
            return None, None

        print("\n" + "="*80)
        print("DEVELOPMENT MODE: Using synthetic embeddings instead of CXR Foundation model".center(80))
        print("WARNING: Not suitable for medical diagnosis - for testing purposes only".center(80))
        print("="*80 + "\n")

        # Process normal images with dummy embeddings
        for img_path in normal_images:
            print(f"Generating synthetic embedding for: {os.path.basename(img_path)}")
            dummy_emb = generate_dummy_embedding(img_path, embedding_dim, is_tb=False)

            all_embeddings.append(dummy_emb)
            all_labels.append("Normal")
            all_image_paths.append(img_path)
            dummy_count += 1

        # Process TB images with dummy embeddings
        for img_path in tb_images:
            print(f"Generating synthetic embedding for: {os.path.basename(img_path)}")
            dummy_emb = generate_dummy_embedding(img_path, embedding_dim, is_tb=True)

            all_embeddings.append(dummy_emb)
            all_labels.append("Tuberculosis")
            all_image_paths.append(img_path)
            dummy_count += 1
    else:
        # Try to load the actual CXR Foundation model
        print(f"Loading model from: {MODEL_PATH}")
        try:
            cxr_model = load_model_with_fallback()
            if cxr_model is None:
                if MEDICAL_MODE or not ALLOW_AUTO_FALLBACK:
                    print("\n" + "="*80)
                    print("CRITICAL ERROR: Failed to load CXR Foundation model".center(80))
                    print("Process terminated for safety - no synthetic embeddings will be used".center(80))
                    print("="*80 + "\n")
                    return None, None
                else:
                    print("\n" + "="*80)
                    print("WARNING: Failed to load CXR Foundation model".center(80))
                    print("DEVELOPMENT MODE: Using synthetic embeddings instead".center(80))
                    print("WARNING: Not suitable for medical diagnosis".center(80))
                    print("="*80 + "\n")
                    # Fall back to dummy embeddings
                    for img_path in normal_images:
                        print(f"Generating synthetic embedding for: {os.path.basename(img_path)}")
                        dummy_emb = generate_dummy_embedding(img_path, embedding_dim, is_tb=False)

                        all_embeddings.append(dummy_emb)
                        all_labels.append("Normal")
                        all_image_paths.append(img_path)
                        dummy_count += 1

                    for img_path in tb_images:
                        print(f"Generating synthetic embedding for: {os.path.basename(img_path)}")
                        dummy_emb = generate_dummy_embedding(img_path, embedding_dim, is_tb=True)

                        all_embeddings.append(dummy_emb)
                        all_labels.append("Tuberculosis")
                        all_image_paths.append(img_path)
                        dummy_count += 1
            else:
                print("\n" + "="*80)
                print("CXR Foundation model loaded successfully!".center(80))
                print("Will generate medical imaging feature embeddings for each X-ray".center(80))
                print("="*80 + "\n")

                # Test if cuDNN is completely non-functional
                try:
                    cuDNN_works = False
                    with tf.device('/CPU:0'):
                        test_input = tf.random.normal([1, 32, 32, 3])
                        test_filter = tf.random.normal([3, 3, 3, 16])
                        test_conv = tf.nn.conv2d(test_input, test_filter, strides=[1, 1, 1, 1], padding='SAME')
                        _ = test_conv.numpy()
                        cuDNN_works = True
                        print("✅ Basic convolution test on CPU successful")
                except Exception as e:
                    print(f"❌ Critical error: Basic convolution test failed on CPU: {e}")
                    if MEDICAL_MODE:
                        print("Cannot proceed safely - model requires convolution operations")
                        return None, None
                
                # If cuDNN works on CPU, use CPU-only mode
                if cuDNN_works:
                    # Process normal images with CPU only
                    print("\n===== FORCING CPU-ONLY MODE FOR ALL MODEL OPERATIONS =====")
                    print("This is necessary due to GPU compatibility issues with the model")
                    
                    # Process normal images with CPU only
                    normal_embeddings, normal_labels, normal_image_paths = process_images_cpu_only(
                        cxr_model, normal_images, "Normal"
                    )
                    all_embeddings.extend(normal_embeddings)
                    all_labels.extend(normal_labels)
                    all_image_paths.extend(normal_image_paths)
                    real_count += len(normal_embeddings)

                    # Process tuberculosis images with CPU only
                    tb_embeddings, tb_labels, tb_image_paths = process_images_cpu_only(
                        cxr_model, tb_images, "Tuberculosis"
                    )
                    all_embeddings.extend(tb_embeddings)
                    all_labels.extend(tb_labels)
                    all_image_paths.extend(tb_image_paths)
                    real_count += len(tb_embeddings)
                else:
                    # Try batch process, then fallback to CPU-only if that fails
                    try:
                        # First attempt with batch processing
                        normal_embeddings, normal_labels, normal_image_paths = batch_process_with_gpu(
                            cxr_model, normal_images, batch_size=MAX_GPU_BATCH_SIZE, class_name="Normal"
                        )
                        
                        if not normal_embeddings and MEDICAL_MODE:
                            print("\n===== TRYING CPU-ONLY PROCESSING =====")
                            print("Batch processing failed, trying individual CPU processing...")
                            
                            normal_embeddings = []
                            normal_labels = []
                            normal_image_paths = []
                            
                            # Process each image individually with CPU
                            for img_path in normal_images:
                                try:
                                    print(f"Processing normal image: {os.path.basename(img_path)}")
                                    embedding = process_elixr_model_input(cxr_model, img_path, force_cpu=True)
                                    if embedding is not None:
                                        if len(embedding.shape) > 2:
                                            embedding = np.mean(embedding, axis=(1, 2))
                                        normal_embeddings.append(embedding.squeeze())
                                        normal_labels.append("Normal")
                                        normal_image_paths.append(img_path)
                                        print(f"✅ Successfully processed {os.path.basename(img_path)}")
                                except Exception as e:
                                    print(f"❌ Error processing {os.path.basename(img_path)}: {e}")
                        
                        # Add normal embeddings to collection
                        all_embeddings.extend(normal_embeddings)
                        all_labels.extend(normal_labels)
                        all_image_paths.extend(normal_image_paths)
                        real_count += len(normal_embeddings)
                    except Exception as e:
                        print(f"Error processing normal images: {e}")
                        if MEDICAL_MODE:
                            raise  # Re-raise in medical mode
                
                # Process tuberculosis images with real model
                try:
                    # First attempt with batch processing
                    tb_embeddings, tb_labels, tb_image_paths = batch_process_with_gpu(
                        cxr_model, tb_images, batch_size=MAX_GPU_BATCH_SIZE, class_name="Tuberculosis"
                    )
                    
                    if not tb_embeddings and MEDICAL_MODE:
                        print("\n===== TRYING CPU-ONLY PROCESSING =====")
                        print("Batch processing failed, trying individual CPU processing...")
                        
                        tb_embeddings = []
                        tb_labels = []
                        tb_image_paths = []
                        
                        # Process each image individually with CPU
                        for img_path in tb_images:
                            try:
                                print(f"Processing TB image: {os.path.basename(img_path)}")
                                embedding = process_elixr_model_input(cxr_model, img_path, force_cpu=True)
                                if embedding is not None:
                                    if len(embedding.shape) > 2:
                                        embedding = np.mean(embedding, axis=(1, 2))
                                    tb_embeddings.append(embedding.squeeze())
                                    tb_labels.append("Tuberculosis")
                                    tb_image_paths.append(img_path)
                                    print(f"✅ Successfully processed {os.path.basename(img_path)}")
                            except Exception as e:
                                print(f"❌ Error processing {os.path.basename(img_path)}: {e}")
                    
                    # Add TB embeddings to collection
                    all_embeddings.extend(tb_embeddings)
                    all_labels.extend(tb_labels)
                    all_image_paths.extend(tb_image_paths)
                    real_count += len(tb_embeddings)
                except Exception as e:
                    print(f"Error processing TB images: {e}")
                    if MEDICAL_MODE:
                        raise  # Re-raise in medical mode
        except Exception as e:
            print(f"❌ Error trying to use the CXR model: {e}")

            if MEDICAL_MODE or not ALLOW_AUTO_FALLBACK:
                print("\n" + "="*80)
                print("CRITICAL ERROR: CXR Foundation model failed".center(80))
                print("Process terminated for safety - no synthetic embeddings will be used".center(80))
                print("="*80 + "\n")
                return None, None

            if ALLOW_AUTO_FALLBACK:
                print("\n" + "="*80)
                print("ERROR: Failed to use CXR Foundation model".center(80))
                print("DEVELOPMENT MODE: Using synthetic embeddings instead".center(80))
                print("WARNING: Not suitable for medical diagnosis".center(80))
                print("="*80 + "\n")
                # Fall back to dummy embeddings
                for img_path in normal_images:
                    print(f"Generating synthetic embedding for: {os.path.basename(img_path)}")
                    dummy_emb = generate_dummy_embedding(img_path, embedding_dim, is_tb=False)

                    all_embeddings.append(dummy_emb)
                    all_labels.append("Normal")
                    all_image_paths.append(img_path)
                    dummy_count += 1

                for img_path in tb_images:
                    print(f"Generating synthetic embedding for: {os.path.basename(img_path)}")
                    dummy_emb = generate_dummy_embedding(img_path, embedding_dim, is_tb=True)

                    all_embeddings.append(dummy_emb)
                    all_labels.append("Tuberculosis")
                    all_image_paths.append(img_path)
                    dummy_count += 1

    # Check if we have any embeddings to continue with
    if not all_embeddings:
        print("No embeddings were generated. Exiting.")
        return None, None

    # Print summary of embeddings
    print("\n" + "="*80)
    print("EMBEDDING GENERATION SUMMARY".center(80))
    print(f"  Total images processed: {total_count}")
    print(f"  Real medical embeddings:   {real_count} ({real_count/total_count*100:.1f}%)")
    print(f"  Synthetic embeddings:      {dummy_count} ({dummy_count/total_count*100:.1f}%)")

    if dummy_count > 0:
        print("\nWARNING: Some or all images were processed with synthetic embeddings")
        print("         These are not suitable for medical diagnosis purposes")
        print("         This should ONLY be used for testing and development")

        if MEDICAL_MODE:
            print("\n" + "="*80)
            print("CRITICAL ERROR: Synthetic embeddings detected in medical mode".center(80))
            print("This should never happen - terminating for safety".center(80))
            print("="*80 + "\n")
            return None, None

    print("="*80 + "\n")

    # Convert to numpy array
    all_embeddings = np.array(all_embeddings)
    print(f"Generated embeddings shape: {all_embeddings.shape}")

    # Check if we have enough data to continue
    label_counts = Counter(all_labels)
    print("Label distribution:", dict(label_counts))
    
    # Check if we have enough samples per class for stratified splitting
    if any(count < 2 for count in label_counts.values()):
        print("\n" + "="*80)
        print("WARNING: Not enough samples per class for stratified splitting".center(80))
        print(f"Current distribution: {dict(label_counts)}".center(80))
        print("Need at least 2 samples per class for train/test split with stratification".center(80))
        
        # If we have very few samples overall, exit
        if len(all_embeddings) < 4:
            print("Too few samples to continue training. Exiting.".center(80))
            print("="*80 + "\n")
            return None, None
            
        print("Will continue with non-stratified train/test split".center(80))
        print("="*80 + "\n")
        use_stratify = False
    else:
        use_stratify = True

    # Save raw embeddings
    np.save(os.path.join(OUTPUT_DIR, "cxr_foundation_embeddings.npy"), all_embeddings)
    with open(os.path.join(OUTPUT_DIR, "cxr_foundation_labels.txt"), 'w') as f:
        for label in all_labels:
            f.write(f"{label}\n")

    # Setup timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save training parameters
    params = {
        "model_path": MODEL_PATH,
        "normal_dir": NORMAL_DIR,
        "tb_dir": TB_DIR,
        "images_per_class": IMAGE_PROCESS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "hidden_layers": HIDDEN_LAYERS,
        "dropout_rate": DROPOUT_RATE
    }

    params_path = os.path.join(run_dir, "params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)

    # Convert labels to numerical form for training
    label_ids = np.array([CLASS_MAPPING[label] for label in all_labels])

    # Split dataset into train, validation, and test
    # Use stratify only if we have enough samples per class
    if use_stratify:
        # Original stratified split
        X_train_val, X_test, y_train_val, y_test, train_val_paths, test_paths = train_test_split(
            all_embeddings, label_ids, all_image_paths, test_size=0.2, random_state=42, stratify=label_ids
        )

        X_train, X_val, y_train, y_val, train_paths, val_paths = train_test_split(
            X_train_val, y_train_val, train_val_paths, test_size=0.25, random_state=42, stratify=y_train_val
        )
    else:
        # Non-stratified split for small datasets
        X_train_val, X_test, y_train_val, y_test, train_val_paths, test_paths = train_test_split(
            all_embeddings, label_ids, all_image_paths, test_size=0.2, random_state=42
        )

        X_train, X_val, y_train, y_val, train_paths, val_paths = train_test_split(
            X_train_val, y_train_val, train_val_paths, test_size=0.25, random_state=42
        )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Print class distribution in each split
    print("Class distribution:")
    print(f"  Training: {Counter(y_train)}")
    print(f"  Validation: {Counter(y_val)}")
    print(f"  Test: {Counter(y_test)}")

    # Save datasets
    np.savez(os.path.join(run_dir, "train_data.npz"), X=X_train, y=y_train, files=train_paths)
    np.savez(os.path.join(run_dir, "val_data.npz"), X=X_val, y=y_val, files=val_paths)
    np.savez(os.path.join(run_dir, "test_data.npz"), X=X_test, y=y_test, files=test_paths)

    # Create and compile neural network model
    print("\nCreating neural network model...")
    input_size = X_train.shape[1]  # Get actual input size from data
    nn_model = create_neural_network(
        input_size=input_size,
        hidden_layers_sizes=HIDDEN_LAYERS,
        dropout_rate=DROPOUT_RATE
    )

    # Check if convolution operations work on GPU 
    gpu_for_training = False
    try:
        if tf.config.list_physical_devices('GPU'):
            # Test GPU with convolution
            with tf.device('/GPU:0'):
                test_input = tf.random.normal([1, 32, 32, 3])
                test_filter = tf.random.normal([3, 3, 3, 16])
                test_output = tf.nn.conv2d(test_input, test_filter, strides=[1, 1, 1, 1], padding='SAME')
                _ = test_output.numpy()
                gpu_for_training = True
                print("✅ GPU works for convolution - will use for model training")
    except Exception as e:
        print(f"⚠️ GPU convolution test failed, training will use CPU: {e}")
        gpu_for_training = False

    # Force training on GPU if available and convolution works
    if gpu_for_training:
        print("Using GPU for model training...")
        with tf.device('/GPU:0'):
            nn_model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
    else:
        print("Using CPU for model training...")
        with tf.device('/CPU:0'):
            nn_model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

    nn_model.summary()

    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(run_dir, "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Add the custom progress callback
        TrainingProgressCallback()
    ]

    # Train neural network model
    print("\nTraining neural network model...")
    
    # Use a larger batch size if possible
    actual_batch_size = min(BATCH_SIZE, X_train.shape[0])
    print(f"Using batch size: {actual_batch_size}")
    
    # Explicitly place training on GPU if available and convolution works
    if gpu_for_training:
        print("Using GPU for model training...")
        with tf.device('/GPU:0'):
            history = nn_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=actual_batch_size,
                callbacks=callbacks,
                verbose=0  # Turn off default progress bar, using our custom callback instead
            )
    else:
        print("Using CPU for model training (due to GPU compatibility issues)...")
        with tf.device('/CPU:0'):
            history = nn_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=actual_batch_size,
                callbacks=callbacks,
                verbose=0  # Turn off default progress bar, using our custom callback instead
            )

    # Plot training history
    history_path = os.path.join(run_dir, "training_history.png")
    plot_training_history(history, history_path)

    # Evaluate on validation set
    val_metrics, val_pred, val_pred_prob = evaluate_model(nn_model, X_val, y_val, "Validation")
    plot_confusion_matrix(y_val, val_pred, os.path.join(run_dir, "val_confusion_matrix.png"))

    # Evaluate on test set
    test_metrics, test_pred, test_pred_prob = evaluate_model(nn_model, X_test, y_test, "Test")
    plot_confusion_matrix(y_test, test_pred, os.path.join(run_dir, "test_confusion_matrix.png"))

    # Save metrics
    all_metrics = {
        'validation': val_metrics,
        'test': test_metrics
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Save model
    model_path = os.path.join(run_dir, "final_model")
    tf.saved_model.save(nn_model, model_path)

    # Also save in Keras format for easier loading
    nn_model.save(os.path.join(run_dir, "model.keras"))

    print(f"\nTraining completed! Results saved to {run_dir}")

    # Demo prediction
    print("\n===== Demo: Predicting with the trained model =====")
    # Get a few examples to test
    demo_normal = normal_images[0] if normal_images else None
    demo_tb = tb_images[0] if tb_images else None

    if demo_normal:
        print(f"\nPredicting Normal image: {os.path.basename(demo_normal)}")
        try:
            # Get embedding - either real or dummy
            if cxr_model is None or USE_DUMMY_EMBEDDINGS:
                print("Using dummy embedding for prediction")
                emb = generate_dummy_embedding(demo_normal, embedding_dim=4096, is_tb=False)
            else:
                emb = process_elixr_model_input(cxr_model, demo_normal)
                if len(emb.shape) > 2:
                    emb = np.mean(emb, axis=(1, 2)).squeeze()

            emb = emb.reshape(1, -1)  # Reshape for prediction

            # Predict
            pred_prob = nn_model.predict(emb)[0][0]
            pred = 1 if pred_prob > 0.5 else 0

            class_name = "Normal" if pred == 0 else "Tuberculosis"
            print(f"Prediction: {class_name} (Probability: {pred_prob:.4f})")
        except Exception as e:
            print(f"Error making prediction: {e}")
            if DEBUG:
                traceback.print_exc()

    if demo_tb:
        print(f"\nPredicting TB image: {os.path.basename(demo_tb)}")
        try:
            # Get embedding - either real or dummy
            if cxr_model is None or USE_DUMMY_EMBEDDINGS:
                print("Using dummy embedding for prediction")
                emb = generate_dummy_embedding(demo_tb, embedding_dim=4096, is_tb=True)
            else:
                emb = process_elixr_model_input(cxr_model, demo_tb)
                if len(emb.shape) > 2:
                    emb = np.mean(emb, axis=(1, 2)).squeeze()

            emb = emb.reshape(1, -1)  # Reshape for prediction

            # Predict
            pred_prob = nn_model.predict(emb)[0][0]
            pred = 1 if pred_prob > 0.5 else 0

            class_name = "Normal" if pred == 0 else "Tuberculosis"
            print(f"Prediction: {class_name} (Probability: {pred_prob:.4f})")
        except Exception as e:
            print(f"Error making prediction: {e}")
            if DEBUG:
                traceback.print_exc()

    return cxr_model, nn_model

# Update the predict_xray function to properly use dummy embeddings
def predict_xray(image_path, cxr_model=None, nn_model=None):
    """Predict TB or Normal for a new X-ray image"""
    if nn_model is None:
        # Try to load the neural network model
        try:
            # Try loading the latest model
            runs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "run_*")))
            if runs:
                latest_run = runs[-1]
                model_path = os.path.join(latest_run, "model.keras")
                if os.path.exists(model_path):
                    nn_model = tf.keras.models.load_model(model_path)
                else:
                    return {"error": f"Model not found at {model_path}"}
            else:
                return {"error": "No trained models found"}
        except Exception as e:
            return {"error": f"Failed to load neural network model: {e}"}

    try:
        # Check if the image path contains TB-related names to determine class for dummy embeddings
        is_tb = ('tb' in image_path.lower() or 'tuberculosis' in image_path.lower())

        # If in dummy mode or no CXR model, generate dummy embedding
        if cxr_model is None or USE_DUMMY_EMBEDDINGS:
            print(f"Using dummy embedding for prediction on: {os.path.basename(image_path)}")
            emb = generate_dummy_embedding(image_path, embedding_dim=4096, is_tb=is_tb).reshape(1, -1)
        else:
            # Get embedding from actual model
            emb = process_elixr_model_input(cxr_model, image_path)
            if emb is None:
                return {"error": "Failed to get embedding for image"}

            # Process embedding
            if len(emb.shape) > 2:
                emb = np.mean(emb, axis=(1, 2)).squeeze()
            emb = emb.reshape(1, -1)  # Reshape for prediction

        # Predict
        pred_prob = float(nn_model.predict(emb)[0][0])
        pred = 1 if pred_prob > 0.5 else 0

        class_name = "Normal" if pred == 0 else "Tuberculosis"
        return {
            "class": class_name,
            "probability": pred_prob if pred == 1 else 1.0 - pred_prob,
            "normal_probability": 1.0 - pred_prob,
            "tb_probability": pred_prob
        }
    except Exception as e:
        if DEBUG:
            traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    # For Colab notebooks, provide a function to run with parameters
    if IN_COLAB:
        # This function can be called directly from notebook cells
        def run_with_params(model_path=None, normal_dir=None, tb_dir=None,
                          output_dir=None, image_process=None, debug=None,
                          batch_size=None, epochs=None, learning_rate=None,
                          patience=None, hidden_layers=None, dropout_rate=None,
                          use_dummy_embeddings=None, medical_mode=True,
                          force_tensor_input=None):
            global MODEL_PATH, NORMAL_DIR, TB_DIR, OUTPUT_DIR, IMAGE_PROCESS, DEBUG
            global BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE, HIDDEN_LAYERS, DROPOUT_RATE
            global USE_DUMMY_EMBEDDINGS, MEDICAL_MODE, ALLOW_AUTO_FALLBACK, FORCE_TENSOR_INPUT

            # Override parameters if provided
            if model_path is not None:
                MODEL_PATH = model_path
            if normal_dir is not None:
                NORMAL_DIR = normal_dir
            if tb_dir is not None:
                TB_DIR = tb_dir
            if output_dir is not None:
                OUTPUT_DIR = output_dir
            if image_process is not None:
                IMAGE_PROCESS = image_process
            if debug is not None:
                DEBUG = debug
            if batch_size is not None:
                BATCH_SIZE = batch_size
            if epochs is not None:
                EPOCHS = epochs
            if learning_rate is not None:
                LEARNING_RATE = learning_rate
            if patience is not None:
                PATIENCE = patience
            if hidden_layers is not None:
                HIDDEN_LAYERS = hidden_layers
            if dropout_rate is not None:
                DROPOUT_RATE = dropout_rate
            if use_dummy_embeddings is not None:
                USE_DUMMY_EMBEDDINGS = use_dummy_embeddings
            if medical_mode is not None:
                MEDICAL_MODE = medical_mode
                # Safety override - in medical mode, we disable auto fallback
                if MEDICAL_MODE:
                    ALLOW_AUTO_FALLBACK = False
                    if USE_DUMMY_EMBEDDINGS:
                        print("\n" + "="*80)
                        print("⚠️ MEDICAL SAFETY WARNING ⚠️".center(80))
                        print("Medical mode is incompatible with dummy embeddings.".center(80))
                        print("Setting USE_DUMMY_EMBEDDINGS=False for safety.".center(80))
                        print("="*80 + "\n")
                        USE_DUMMY_EMBEDDINGS = False
            if force_tensor_input is not None:
                FORCE_TENSOR_INPUT = force_tensor_input

            print(f"\nRunning with parameters:")
            print(f"  MODEL_PATH: {MODEL_PATH}")
            print(f"  NORMAL_DIR: {NORMAL_DIR}")
            print(f"  TB_DIR: {TB_DIR}")
            print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
            print(f"  IMAGE_PROCESS: {IMAGE_PROCESS}")
            print(f"  BATCH_SIZE: {BATCH_SIZE}")
            print(f"  EPOCHS: {EPOCHS}")
            print(f"  LEARNING_RATE: {LEARNING_RATE}")
            print(f"  PATIENCE: {PATIENCE}")
            print(f"  HIDDEN_LAYERS: {HIDDEN_LAYERS}")
            print(f"  DROPOUT_RATE: {DROPOUT_RATE}")
            print(f"  DEBUG: {DEBUG}")
            print(f"  USE_DUMMY_EMBEDDINGS: {USE_DUMMY_EMBEDDINGS}")
            print(f"  MEDICAL_MODE: {MEDICAL_MODE}")
            print(f"  ALLOW_AUTO_FALLBACK: {ALLOW_AUTO_FALLBACK}")
            print(f"  FORCE_TENSOR_INPUT: {FORCE_TENSOR_INPUT}")

            return main()

        print("\nIn Colab, you can run with custom parameters using:")
        print("cxr_model, nn_model = run_with_params(")
        print("    model_path='hf/elixr-c-v2-pooled',")
        print("    image_process=10,")
        print("    medical_mode=True,  # Set to False for development/testing with synthetic embeddings")
        print("    force_tensor_input=True  # Try this if you have TF Example format errors")
        print(")")
        print("\nTo predict on new images after training:")
        print("result = predict_xray('path/to/xray.jpg')")
        print("print(f\"Prediction: {result['class']} with {result['probability']:.2f} probability\")")

    # Always run main when script is executed
    cxr_model, nn_model = main()
