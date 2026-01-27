"""
Dense Neural Network (MNIST) for Kubernetes API.

TensorFlow/Keras dense neural network for MNIST digit classification.
Adapted from dense_network_standalone.py.
"""
import gc
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Configuration constants
DEFAULT_NUM_SAMPLES = 20000
DEFAULT_NUM_LAYERS = 2
DEFAULT_UNITS = 128
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TARGET_ACCURACY = 0.90
RESOURCES_DIR = Path(__file__).parent.parent / "resources"
MNIST_FILENAME = "mnist.npz"
DATA_LOAD_BATCH_SIZE = 2000


def _get_mnist_metadata(num_samples: int) -> Dict[str, Any]:
    """Read MNIST dataset metadata (shapes) without loading data into memory."""
    start_time = time.perf_counter()

    path = RESOURCES_DIR / MNIST_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"MNIST dataset not found: {path}. Please place '{MNIST_FILENAME}' in the resources folder."
        )

    with np.load(path) as data:
        num_train = min(num_samples, data["x_train"].shape[0])
        num_test = data["x_test"].shape[0]

    load_time = time.perf_counter() - start_time

    return {
        "path": str(path),
        "num_train": num_train,
        "num_test": num_test,
        "load_time": load_time,
    }


def _create_train_dataset(
    metadata: Dict[str, Any], train_batch_size: int
) -> Tuple["tf.data.Dataset", "tf.data.Dataset", int, int]:
    """Create streaming train and validation datasets from disk."""
    import tensorflow as tf

    path = metadata["path"]
    total = metadata["num_train"]
    num_val = int(total * VALIDATION_SPLIT)
    num_train = total - num_val

    def _make_generator(start: int, end: int):
        def generator():
            for offset in range(start, end, DATA_LOAD_BATCH_SIZE):
                chunk_end = min(offset + DATA_LOAD_BATCH_SIZE, end)
                with np.load(path) as data:
                    x = data["x_train"][offset:chunk_end].astype("float32") / 255.0
                    y = data["y_train"][offset:chunk_end]
                x = x.reshape(-1, 28 * 28)
                for i in range(len(x)):
                    yield x[i], y[i]
        return generator

    output_sig = (
        tf.TensorSpec(shape=(784,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.uint8),
    )

    train_ds = (
        tf.data.Dataset.from_generator(_make_generator(0, num_train), output_signature=output_sig)
        .batch(train_batch_size)
        .prefetch(1)
    )
    val_ds = (
        tf.data.Dataset.from_generator(_make_generator(num_train, total), output_signature=output_sig)
        .batch(train_batch_size)
        .prefetch(1)
    )

    return train_ds, val_ds, num_train, num_val


def _evaluate_batched(
    model, metadata: Dict[str, Any], batch_size: int
) -> float:
    """Evaluate model on test data using streaming batches."""
    import tensorflow as tf

    path = metadata["path"]
    num_test = metadata["num_test"]

    def generator():
        for offset in range(0, num_test, DATA_LOAD_BATCH_SIZE):
            chunk_end = min(offset + DATA_LOAD_BATCH_SIZE, num_test)
            with np.load(path) as data:
                x = data["x_test"][offset:chunk_end].astype("float32") / 255.0
                y = data["y_test"][offset:chunk_end]
            x = x.reshape(-1, 28 * 28)
            for i in range(len(x)):
                yield x[i], y[i]

    output_sig = (
        tf.TensorSpec(shape=(784,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.uint8),
    )

    test_ds = (
        tf.data.Dataset.from_generator(generator, output_signature=output_sig)
        .batch(batch_size)
        .prefetch(1)
    )

    _, accuracy = model.evaluate(test_ds, verbose=0)
    return accuracy


def _build_model(num_layers: int, units: int):
    """Build the Dense neural network."""
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),  # Flattened 28x28
    ])

    # Add configured number of Dense layers
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(units, activation='relu'))

    # Output layer for 10 classes
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def dense_network(
    num_samples: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_layers: Optional[int] = None,
    units: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run Dense Neural Network benchmark on MNIST.

    Args:
        num_samples: Number of training samples (default: 20000)
        epochs: Training epochs (default: 5)
        batch_size: Batch size (default: 32)
        num_layers: Number of hidden layers (default: 2)
        units: Units per layer (default: 128)

    Returns:
        Dictionary with results:
        - execution_time: Training time in seconds
        - accuracy: Final test accuracy
        - loss: Final validation loss
        - training_samples: Number of training samples
        - epochs: Epochs completed
        - target_reached: Whether target accuracy (90%) was reached
        - data_load_time: Time to load dataset
        - success: Boolean indicating success
    """
    import tensorflow as tf

    # Apply defaults
    if num_samples is None:
        num_samples = DEFAULT_NUM_SAMPLES
    if epochs is None:
        epochs = DEFAULT_EPOCHS
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if num_layers is None:
        num_layers = DEFAULT_NUM_LAYERS
    if units is None:
        units = DEFAULT_UNITS

    try:
        # Phase 1 - Metadata (negligible memory)
        metadata = _get_mnist_metadata(num_samples)
        load_time = metadata["load_time"]

        # Phase 2 - Build model
        model = _build_model(num_layers, units)

        # Phase 3 - Train with streaming data
        train_ds, val_ds, num_train, num_val = _create_train_dataset(metadata, batch_size)

        start_time = time.perf_counter()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0
        )

        execution_time = time.perf_counter() - start_time

        del train_ds, val_ds
        gc.collect()

        # Phase 4 - Evaluate with streaming test data
        test_accuracy = _evaluate_batched(model, metadata, batch_size)

        # Get final metrics from validation
        final_loss = history.history['val_loss'][-1]

        del history
        gc.collect()

        # Phase 5 - Cleanup
        result = {
            "execution_time": float(execution_time),
            "accuracy": float(test_accuracy),
            "loss": float(final_loss),
            "training_samples": int(num_train),
            "epochs": int(epochs),
            "target_reached": bool(test_accuracy >= TARGET_ACCURACY),
            "data_load_time": float(load_time),
            "success": True
        }

        del model
        tf.keras.backend.clear_session()
        gc.collect()

        return result

    except Exception as e:
        try:
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception:
            pass

        return {
            "execution_time": 0.0,
            "accuracy": 0.0,
            "loss": 0.0,
            "training_samples": 0,
            "epochs": 0,
            "target_reached": False,
            "data_load_time": 0.0,
            "success": False,
            "error_message": str(e)
        }
