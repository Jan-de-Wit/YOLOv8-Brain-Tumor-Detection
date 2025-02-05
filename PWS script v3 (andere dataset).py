import cv2
from ultralytics import YOLO
import os
from pathlib import Path
import yaml
from roboflow import Roboflow
import torch
import gc
import albumentations as A
import logging
import argparse
import platform
import psutil
import numpy as np
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training.log'),
        logging.StreamHandler()
    ]
)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train tumor detection model')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint', type=str,
                        help='Specific checkpoint to resume from (optional)')
    return parser.parse_args()


def process_single_image(img_file, src_path, dst_path, label_src_path, label_dst_path, transform):
    try:
        # Convert to Path objects if they aren't already
        img_file = Path(img_file)
        src_path = Path(src_path)
        dst_path = Path(dst_path)
        label_src_path = Path(label_src_path)
        label_dst_path = Path(label_dst_path)

        # Setup paths
        img_path = src_path / img_file.name
        output_path = dst_path / img_file.name
        label_file = img_file.stem + '.txt'
        label_src = label_src_path / label_file
        label_dst = label_dst_path / label_file

        # Skip if already processed
        if output_path.exists() and label_dst.exists():
            return True

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            logging.warning(f"Could not read image: {img_path}")
            return False

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        transformed = transform(image=image)
        processed_image = transformed['image']

        # Convert back to BGR for saving
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        # Save processed image
        cv2.imwrite(str(output_path), processed_image)

        # Copy label file if it exists
        if label_src.exists():
            shutil.copy2(str(label_src), str(label_dst))

        return True

    except Exception as e:
        logging.error(f"Error processing {img_file}: {str(e)}")
        return False


def process_batch(batch_files, src_path, dst_path, label_src_path, label_dst_path, transform):
    return [process_single_image(f, src_path, dst_path, label_src_path, label_dst_path, transform)
            for f in batch_files]


def preprocess_dataset():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from tqdm import tqdm

    # Determine optimal number of workers
    num_workers = min(multiprocessing.cpu_count(), 8)

    transform = A.Compose([
        A.MedianBlur(blur_limit=3, p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(10, 10), p=1.0),
        A.Sharpen(alpha=(0.25, 0.35), lightness=(0.85, 1.0), p=0.9),
        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.2,
            p=0.9
        ),
        A.RandomGamma(gamma_limit=(95, 105), p=0.6),
    ])

    # Define paths
    dataset_dir = "multi-tumor-1"
    processed_dir = "processed_dataset"
    splits = [
        ("train",
         os.path.join(dataset_dir, "train/images"),
         os.path.join(processed_dir, "train/images"),
         os.path.join(dataset_dir, "train/labels"),
         os.path.join(processed_dir, "train/labels")
         ),
        ("valid",
         os.path.join(dataset_dir, "valid/images"),
         os.path.join(processed_dir, "valid/images"),
         os.path.join(dataset_dir, "valid/labels"),
         os.path.join(processed_dir, "valid/labels")
         ),
        ("test",
         os.path.join(dataset_dir, "test/images"),
         os.path.join(processed_dir, "test/images"),
         os.path.join(dataset_dir, "test/labels"),
         os.path.join(processed_dir, "test/labels")
         )
    ]

    # Create output directories
    for _, _, dst_path, _, label_dst_path in splits:
        os.makedirs(dst_path, exist_ok=True)
        os.makedirs(label_dst_path, exist_ok=True)

    # Process each split
    total_processed = 0
    total_images = sum(len([f for f in os.listdir(src_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                       for _, src_path, _, _, _ in splits)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for split_name, src_path, dst_path, label_src_path, label_dst_path in splits:
            image_files = [f for f in os.listdir(src_path)
                           if f.endswith(('.jpg', '.jpeg', '.png'))]

            # Skip if no images to process
            if not image_files:
                continue

            # Create batches
            batch_size = 32  # Adjust based on your system's memory
            batches = [image_files[i:i + batch_size]
                       for i in range(0, len(image_files), batch_size)]

            # Process batches in parallel
            logging.info(
                f"Processing {split_name} split with {len(image_files)} images")
            futures = []

            with tqdm(total=len(image_files), desc=f"Processing {split_name}") as pbar:
                for batch in batches:
                    future = executor.submit(
                        process_batch, batch, src_path, dst_path, label_src_path, label_dst_path, transform)
                    future.add_done_callback(lambda p: pbar.update(len(batch)))
                    futures.append(future)

                # Wait for all futures to complete
                results = []
                for future in futures:
                    results.extend(future.result())

                total_processed += sum(results)

    logging.info(
        f"Dataset preprocessing complete. Successfully processed {total_processed}/{total_images} images")
    return processed_dir


class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
                weights.append(1)
                continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """

        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))


def setup_weighted_dataloader():
    # Store original dataset class
    original_dataset = build.YOLODataset

    # Replace with our weighted version
    build.YOLODataset = YOLOWeightedDataset

    return original_dataset  # Return for restoration if needed


class HardwareConfig:
    def __init__(self):
        self.device, self.device_name = self._detect_device()
        self.config = self._get_optimal_config()
        self._log_system_info()

    def _detect_device(self):
        """Detect available hardware and return optimal device."""
        if torch.cuda.is_available():
            device = 'cuda'
            device_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            device = 'mps'
            device_name = 'Apple Silicon'
        else:
            device = 'cpu'
            device_name = 'CPU'
        return device, device_name

    def _get_optimal_config(self):
        """Get optimal training configuration based on hardware."""
        base_config = {
            'device': self.device,
            'imgsz': 640,
            'optimizer': 'AdamW',
            'cos_lr': True,
            'patience': 25,
            'epochs': 100,

            'cache': 'disk',
            'lr0': 0.0005,
            'lrf': 0.0001,
            'warmup_epochs': 6,
            'warmup_momentum': 0.8,
            'weight_decay': 0.03,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.1,

            'degrees': 5,
            'translate': 0.1,
            'scale': 0.1,
            'shear': 0,
            'perspective': 0,
        }

        # Hardware-specific configurations
        if self.device == 'cuda':
            cuda_config = {
                'batch': 14,
                'workers': 4,
            }
            base_config.update(cuda_config)

        elif self.device == 'mps':
            mps_config = {
                'batch': 8,
                'workers': 3,
            }
            base_config.update(mps_config)

        else:  # CPU fallback
            cpu_config = {
                'batch': 4,
                'workers': 2,
            }
            base_config.update(cpu_config)

        return base_config

    def _log_system_info(self):
        """Log detailed system information."""
        memory = psutil.virtual_memory()
        logging.info("\n=== System Information ===")
        logging.info(f"Device: {self.device_name}")
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Python: {platform.python_version()}")
        logging.info(f"PyTorch: {torch.__version__}")
        logging.info(f"Available Memory: {memory.available / (1024**3):.1f}GB")

        if self.device == 'cuda':
            logging.info(f"CUDA Version: {torch.version.cuda}")
            logging.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")

        logging.info("\n=== Training Configuration ===")
        for key, value in self.config.items():
            logging.info(f"{key}: {value}")
        logging.info("========================\n")


class TumorDetectionTrainer:
    def __init__(self, resume=False, checkpoint=None):
        self.work_dir = Path(os.getcwd())
        self.dataset_dir = self.work_dir / 'Multi-Tumor-1'
        self.hardware = HardwareConfig()
        self.resume = resume
        self.checkpoint = checkpoint
        self.data_yaml_path = self.dataset_dir / "data.yaml"

    def clear_memory(self):
        """Clear memory based on device type."""
        gc.collect()
        if self.hardware.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.hardware.device == 'mps':
            torch.mps.empty_cache()

    def setup_dataset(self):
        """
        Downloads the dataset from Roboflow.
        """
        try:
            logging.info("Downloading dataset...")
            rf = Roboflow(api_key="mjlzWzF2MLR8VQMK6qq6")
            project = rf.workspace(
                "universiti-kebangsaan-malaysia-qbroi").project("multi-tumor")
            version = project.version(1)

            # Download dataset and get the Dataset object
            dataset = version.download("yolov8")

            # The dataset is downloaded to the current working directory
            self.dataset_dir = Path.cwd() / "multi-tumor-1"
            logging.info(
                f"Dataset downloaded successfully to {self.dataset_dir}")
            return True
        except Exception as e:
            logging.error(f"Dataset setup failed: {e}")
            return False

    def update_yaml_config(self):
        """Update the YAML configuration file with dataset paths."""
        try:
            # Use absolute paths
            workspace_path = Path(os.getcwd())
            processed_dataset = workspace_path / "processed_dataset"
            self.data_yaml_path = processed_dataset / "data.yaml"

            # Create processed dataset directory if it doesn't exist
            processed_dataset.mkdir(exist_ok=True)

            # Load existing data if yaml exists, otherwise create new
            data = {}
            if self.data_yaml_path.exists():
                with open(self.data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)

            # Update paths with absolute paths
            data.update({
                "path": str(processed_dataset),
                "train": str(processed_dataset / "train/images"),
                "val": str(processed_dataset / "valid/images"),
                "test": str(processed_dataset / "test/images"),
                "nc": 3,
                "names": ['glioma', 'meningioma', 'pituitary']
            })

            # Save updated yaml
            with open(self.data_yaml_path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False)

            logging.info(
                f"YAML configuration updated successfully at {self.data_yaml_path}")
            return self.data_yaml_path

        except Exception as e:
            logging.error(f"YAML configuration failed: {str(e)}")
            return None

    def find_latest_checkpoint(self):
        """
        Finds the latest checkpoint in the runs directory.
        """
        runs_dir = self.work_dir / 'runs' / 'detect'
        if not runs_dir.exists():
            return None

        train_runs = [d for d in runs_dir.iterdir() if d.is_dir()
                      and d.name.startswith('train')]
        if not train_runs:
            return None

        latest_run = max(train_runs, key=lambda x: x.stat().st_mtime)
        weights_dir = latest_run / 'weights'
        if not weights_dir.exists():
            return None

        weights_files = list(weights_dir.glob('*.pt'))
        if not weights_files:
            return None

        return str(max(weights_files, key=lambda x: x.stat().st_mtime))

    def train_model(self, data_yaml_path):
        """
        Trains the model.
        """
        dataset = setup_weighted_dataloader()

        try:
            self.clear_memory()
            model_path = self.checkpoint if self.checkpoint else 'yolov8s.pt'
            model = YOLO(model_path)

            # Get hardware-optimized training args
            training_args = self.hardware.config

            # Add dataset-specific args
            training_args.update({
                'data': str(data_yaml_path),
                'resume': self.resume,
            })

            results = model.train(**training_args)
            return results

        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("Memory error occurred")
                self.clear_memory()
                # Try to recover with smaller batch size
                training_args['batch'] = training_args['batch'] // 2
                logging.info(
                    f"Retrying with batch size: {training_args['batch']}")
                results = model.train(**training_args)
                return results
            logging.error(f"Training error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error during training: {e}")
            return None
        finally:
            build.YOLODataset = dataset


def main():
    args = parse_args()
    trainer = TumorDetectionTrainer(
        resume=args.resume,
        checkpoint=args.checkpoint
    )

    if not trainer.setup_dataset():
        return

    data_yaml_path = trainer.update_yaml_config()
    if not data_yaml_path:
        return

    logging.info("Preprocessing dataset...")
    preprocess_dataset()
    logging.info("Dataset preprocessing complete")

    results = trainer.train_model(data_yaml_path)

    if results:
        logging.info("Training completed successfully")
    else:
        logging.warning("Training failed or was interrupted")


if __name__ == "__main__":
    main()
