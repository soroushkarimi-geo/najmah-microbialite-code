"""Shared project settings: paths, microfacies scheme, facies belts and model
hyperparameters.

Every classification script imports ``Config`` from here, so the XGBoost
settings and the list of microfacies are the same in all of them.
"""
from pathlib import Path


class Config:
    """Project-wide constants, grouped as class attributes.

    Attributes:
        BASE_DIR: Repository root (the folder that holds this file).
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR,
            OUTPUTS_DIR: Default working folders under ``BASE_DIR``.
        SECTIONS: Measured sections, with their sample-code prefix and the
            range of sample numbers that were imaged.
        IMAGE_TYPES: Image modes in the image archive.
        MICROFACIES: Najmah Formation microfacies codes known to the
            classifier (MF-A is excluded, see the comment below).
        FACIES_BELTS: Facies belts and the litho-/microfacies assigned to each,
            after Bayet-Goll et al. (2022).
        ENVIRONMENTAL_ZONE: Facies belt of each microfacies.
        XGBOOST_PARAMS: Hyperparameters shared by every XGBoost model in the
            project. ``num_class`` is overwritten at run time with the number
            of classes actually present.
        AUGMENTATION: Image augmentation settings (rotations, brightness and
            contrast ranges).
        SEQUENCE_FRAMEWORK: Sequence-stratigraphic model used to name the
            boundaries (Embry and Johannson, 1992).
    """

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = BASE_DIR / 'models'
    LOGS_DIR = BASE_DIR / 'logs'
    OUTPUTS_DIR = BASE_DIR / 'outputs'

    # Measured sections and sample number ranges
    SECTIONS = {
        'dudan': {'code': 'SD', 'range': (87, 145)},
        'kezi': {'code': 'SK', 'range': (83, 87)}
    }

    IMAGE_TYPES = ['PPL', 'XPL', 'Binocular']

    # Najmah Formation microfacies used by the classifier.
    # MF-A has only 3 specimens, too few for leave-one-specimen-out, so it is
    # left out here (it is still part of the facies model below).
    # MF-F and MF-G names follow our own sample-by-sample review.
    MICROFACIES = [
        # 'MF-A',   # Dolostone - Lime Mudstone
        'MF-B',   # Stromatolite (Dolo)Boundstone = Class 1
        'MF-C',   # Bioclastic Calcisphere Mudstone-Wackestone-Packstone
        'MF-D',   # Peloid Oncoid Microbial (Dolo)Boundstone
        'MF-E',   # Thrombolite = Class 2
        'MF-F',   # Mollusk Wackestone
        'MF-G',   # Radiolarian Wackestone/Radiolarite
    ]

    # Facies belts after Bayet-Goll et al. (2022, Geological Journal 57:2058-2093).
    # "Dudan" in this project = "Dawan" in that paper.
    # Stromatolite (F6b) and thrombolite (F6c) both belong to facies association F6,
    # shallow subtidal - lower intertidal, so MF-B and MF-E share one belt.
    # LF-A is the brecciated, exposure-reworked equivalent of MF-B.
    FACIES_BELTS = {
        'Deep_Subtidal': {
            'lithofacies': ['LF-B'],
            'microfacies': [],
        },
        'Shallow_Subtidal-Intertidal': {
            'lithofacies': [],
            'microfacies': ['MF-B', 'MF-C', 'MF-D', 'MF-E'],
        },
        'Intertidal-Supratidal': {
            'lithofacies': ['LF-A'],  # supratidal
            'microfacies': ['MF-A'],
        },
    }

    ENVIRONMENTAL_ZONE = {
        'MF-A': 'Intertidal-Supratidal',
        'MF-B': 'Shallow_Subtidal-Intertidal',
        'MF-C': 'Shallow_Subtidal-Intertidal',
        'MF-D': 'Shallow_Subtidal-Intertidal', 'MF-E': 'Shallow_Subtidal-Intertidal',
    }

    # XGBoost hyperparameters
    # num_class is reset to the number of classes present in train_xgboost_cv.py
    XGBOOST_PARAMS = {
        'max_depth': 8,
        'n_estimators': 300,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softmax',
        'num_class': 6,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }

    AUGMENTATION = {
        'rotation_angles': [0, 90, 180, 270],
        'brightness_range': (0.95, 1.05),
        'contrast_range': (0.95, 1.05)
    }

    # Sequence stratigraphy framework
    SEQUENCE_FRAMEWORK = 'Embry_Johannson_1992'

    @classmethod
    def create_directories(cls):
        """Create the default data, model, log and output folders if missing.

        Existing folders are left untouched.
        """
        for dir_path in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
                         cls.MODELS_DIR, cls.LOGS_DIR, cls.OUTPUTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
