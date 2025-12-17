try:
    import horizon_driving_dataset

    HORIZON_DRIVING_DATASET_AVAILABLE = True
except ImportError:
    HORIZON_DRIVING_DATASET_AVAILABLE = False
