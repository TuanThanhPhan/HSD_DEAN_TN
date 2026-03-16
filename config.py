BATCH_SIZE = 64
LR = 5e-6
MAX_LEN = 80
EPOCHS = 30
PATIENCE = 5

MODEL_NAME = "vinai/phobert-base"
MODEL_TYPE = "hybrid"   # phobert | visobert | hybrid

TRAIN_PATH = "data/processed/train.csv"
DEV_PATH = "data/processed/dev.csv"
TEST_PATH = "data/processed/test.csv"

SAVE_DIR = "checkpoints"

MODEL_FILE = "best_model.pt"
CHAR_VOCAB_FILE = "char_vocab.pkl"