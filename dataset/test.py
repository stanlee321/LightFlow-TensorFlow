from dataset import load_batch
from dataset_configs import FLYING_CHAIRS_DATASET_CONFIG


if __name__ == "__main__":
    print(FLYING_CHAIRS_DATASET_CONFIG["PATHS"]["sample"])
    input_a, input_b, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, "sample", 1)
    print(input_a)