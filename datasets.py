import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


# Custom Dataset for Triplet Sampling
class TripletVisA(Dataset):
    def __init__(self, path, split="train", val_split=0.2, seed=42):
        """
        Dataset loader for the VisA dataset to generate triplets (anchor, positive, negative).

        Args:
            path (str): Path to the VisA dataset root directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.categories = self._get_categories(path)
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((256, 256)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.seed = seed
        self.split = split
        self.val_split = val_split
        all_data = self._generate_triplets(path)
        self.data = self._split_data(all_data)

    def _get_categories(self, path):
        """Get all category directories under the given path."""
        return [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]

    def _load_images_from_folder(self, folder):
        """Load all image file paths from a given folder."""
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def _generate_triplets(self, path):
        """Generate triplets (anchor, positive, negative) for all categories."""
        triplets = []
        for category in self.categories:
            normal_folder = os.path.join(category, "Data", "Images", "Normal")
            anomaly_folder = os.path.join(category, "Data", "Images", "Anomaly")

            if not os.path.exists(normal_folder) or not os.path.exists(anomaly_folder):
                continue

            normal_images = self._load_images_from_folder(normal_folder)
            anomalous_images = self._load_images_from_folder(anomaly_folder)

            if not anomalous_images or len(normal_images) < 2:
                continue

            # Create triplets by iterating over anomalous images
            for anomaly in anomalous_images:
                # Randomly sample 2 different normal images
                anchor, positive = random.sample(normal_images, 2)
                triplets.append((anchor, positive, anomaly))

        return triplets

    def _split_data(self, all_triplets):
        """Split data into train and validation sets."""
        random.seed(self.seed)
        total_size = len(all_triplets)
        val_size = int(total_size * self.val_split)

        if self.split == "train":
            return all_triplets[val_size:]
        elif self.split == "val":
            return all_triplets[:val_size]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.data[idx]

        # Load images
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        # Apply transforms if specified
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
