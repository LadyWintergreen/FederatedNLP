import numpy as np
import torch



from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "distilbert-base-uncased"
