import os
import shutil
import tensorflow as tf

def clear_keras_cache():
    """Clear Keras cache to fix weight loading issues"""
    
    # Get Keras cache directory
    keras_dir = os.path.expanduser('~/.keras')
    
    # Also check Windows AppData location
    if os.name == 'nt':  # Windows
        keras_dir_alt = os.path.join(os.environ.get('USERPROFILE', ''), '.keras')
        if os.path.exists(keras_dir_alt):
            keras_dir = keras_dir_alt
    
    print(f"Looking for Keras cache at: {keras_dir}")
    
    if os.path.exists(keras_dir):
        try:
            print("Clearing Keras cache...")
            shutil.rmtree(keras_dir)
            print("✓ Keras cache cleared successfully")
        except Exception as e:
            print(f"✗ Error clearing cache: {e}")
            print("You may need to run as administrator")
    else:
        print("No Keras cache found")
    
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    print("✓ TensorFlow session cleared")
    
    print("\nNow try running your training script again.")

if __name__ == "__main__":
    clear_keras_cache()