"""Train wav2vec for autism classification (on both stories and triangles task - probably do one for each)
TODO
- try the custom Wav2Vec2Processor and CTCTrainer and DataCollatorCTCWithPaddingKlaam
- experiment with parameters
"""

# from platform import processor
import datasets
import numpy as np
import torch
import torchaudio
import os
from datasets import load_dataset

from typing import Union, Optional, Dict, List
# from src.trainer import CTCTrainer
from dataclasses import dataclass, field
class CustomWav2Vec2Processor:
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.
    :class:`~transformers.Wav2Vec2Processor` offers all the functionalities of
    :class:`~transformers.Wav2Vec2FeatureExtractor` and :class:`~transformers.Wav2Vec2CTCTokenizer`. See the docstring
    of :meth:`~transformers.Wav2Vec2Processor.__call__` and :meth:`~transformers.Wav2Vec2Processor.decode` for more
    information.
    Args:
        feature_extractor (:obj:`Wav2Vec2FeatureExtractor`):
            An instance of :class:`~transformers.Wav2Vec2FeatureExtractor`. The feature extractor is a required input.
        tokenizer (:obj:`Wav2Vec2CTCTokenizer`):
            An instance of :class:`~transformers.Wav2Vec2CTCTokenizer`. The tokenizer is a required input.
    """

    def __init__(self, feature_extractor):

        self.feature_extractor = feature_extractor
        self.current_processor = self.feature_extractor


    def save_pretrained(self, save_directory):
        """
        Save a Wav2Vec2 feature_extractor object and Wav2Vec2 tokenizer object to the directory ``save_directory``, so
        that it can be re-loaded using the :func:`~transformers.Wav2Vec2Processor.from_pretrained` class method.
        .. note::
            This class method is simply calling
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.save_pretrained` and
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.save_pretrained`. Please refer to the
            docstrings of the methods above for more information.
        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """

        self.feature_extractor.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a :class:`~transformers.Wav2Vec2Processor` from a pretrained Wav2Vec2 processor.
        .. note::
            This class method is simply calling Wav2Vec2FeatureExtractor's
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.from_pretrained` and
            Wav2Vec2CTCTokenizer's :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
            Please refer to the docstrings of the methods above for more information.
        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:
                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :meth:`~transformers.SequenceFeatureExtractor.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/feature_extraction_config.json``.
            **kwargs
                Additional keyword arguments passed along to both :class:`~transformers.SequenceFeatureExtractor` and
                :class:`~transformers.PreTrainedTokenizer`
        """
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor)

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        :meth:`~transformers.Wav2Vec2FeatureExtractor.__call__` and returns its output. If used in the context
        :meth:`~transformers.Wav2Vec2Processor.as_target_processor` this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's :meth:`~transformers.Wav2Vec2CTCTokenizer.__call__`. Please refer to the doctsring of
        the above two methods for more information.
        """
        return self.current_processor(*args, **kwargs)

    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        :meth:`~transformers.Wav2Vec2FeatureExtractor.pad` and returns its output. If used in the context
        :meth:`~transformers.Wav2Vec2Processor.as_target_processor` this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's :meth:`~transformers.Wav2Vec2CTCTokenizer.pad`. Please refer to the docstring of the
        above two methods for more information.
        """
        return self.current_processor.pad(*args, **kwargs)

@dataclass
class DataCollatorCTCWithInputPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: CustomWav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch


from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    EvalPrediction,
    TrainingArguments,
    Trainer, Wav2Vec2Processor)

# set constants
WORKING_DIR = "/usr/src/temp/"
#WORKING_DIR = "./"
DATA_FOLDER = "/usr/src/temp/data/"
#DATA_FOLDER = "./data/"
MODEL_NAME = WORKING_DIR + "models/wav2vec2-base"
OUTPUT_DIR = os.path.join("models", )
TRAIN = os.path.join(DATA_FOLDER ,"train.json.csv")
VALIDATION = os.path.join(DATA_FOLDER, "val.json.csv")

FREEZE_ENCODER = True
FREEZE_BASE_MODEL = False
# specify input/label columns
INPUT_COL = "audio"
LABEL_COL = "label"
# training params
EPOCHS = 10
LEARNING_RATE = 1e-3  # 3e-3
BATCH_SIZE = 8

# model params              # default
ATTENTION_DROPOUT = 0.1  # 0.1
HIDDEN_DROPOUT = 0.1  # 0.1
FEAT_PROJ_DROPOUT = 0.0  # 0.1
MASK_TIME_PROB = 0.05  # 0.075
LAYERDROP = 0.1  # 0.1
GRADIENT_CHECKPOINTING = True  # False
CTC_LOSS_REDUCTION = "sum"  # "sum"   - try "mean"

params = {"attention_dropout": ATTENTION_DROPOUT,
          "hidden_dropout": HIDDEN_DROPOUT,
          "feat_proj_dropout": FEAT_PROJ_DROPOUT,
          "mask_time_prob": MASK_TIME_PROB,
          "layerdrop": LAYERDROP,
          "gradient_checkpointing": GRADIENT_CHECKPOINTING,
          "ctc_loss_reduction": CTC_LOSS_REDUCTION}


def cut_audio( signal, truncate_type, length):
    l = signal.shape[1]
    if l <= length:
        return signal
    if truncate_type == "start":
        return signal[:, :length]
    elif truncate_type == "end":
        return signal[:, -length:]
    else:
        m = int(l / 2)

        return signal[:, m - int(length / 2):m + int(length / 2) ]


def right_pad_if_necessary( signal, num_samples):
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

import os
def mix_down_if_necessary( signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal
# Preprocessing functions
def speech_file_to_array(path):

        "resample audio to match what the model expects (16000 khz)"
        signal, sampling_rate = torchaudio.load(path)
        signal = cut_audio(signal, truncate_type="mm", length = sampling_rate * 20)
        signal = right_pad_if_necessary(signal, sampling_rate* 20)
        signal = mix_down_if_necessary(signal)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(signal).squeeze().numpy()
        #print(len(speech))
        return speech



def label_to_id(label, label_list):
    "map label to id int"

    return label_list.index(label)


def preprocess(batch):
    "preprocess hf dataset/load data"
    speech_list = []
    labels = []
    for index,path in enumerate(batch[INPUT_COL]):
        if os.path.exists(path) :
            try:
                signal = speech_file_to_array(path)
                speech_list.append(signal)
                labels.append(batch[LABEL_COL][index])
            except Exception as ex:
                print(ex)
                print("[EX]", index, path)
        else:
            print(index, path)
 #   speech_list = [speech_file_to_array(path) for path in batch[INPUT_COL]]
 #   labels = [label_to_id(label, label_list) for label in batch[LABEL_COL]]

    out = processor(speech_list, sampling_rate=target_sampling_rate)
    out["labels"] = list(labels)

    return out


# which metrics to compute for evaluation
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


if __name__ == "__main__":

    ################### LOAD DATASETS
    #######
    ####
    # load datasets
    data_files = {
        "train": TRAIN,
        "validation": VALIDATION
    }

    print("[INFO] Loading dataset...")
    dataset = load_dataset("csv", data_files=data_files, delimiter=",")
    train = dataset["train"]
    val = dataset["validation"]

    # get labels and num labels
    label_list = train.unique(LABEL_COL)
    # sorting for determinism
    label_list.sort()
    num_labels = len(label_list)
    print(num_labels)
    # Load feature extractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    # need this parameter for preprocessing to resample audio to correct sampling rate
    target_sampling_rate = processor.sampling_rate

    # preprocess datasets
    print("[INFO] Preprocessing dataset...")



    if os.path.exists(DATA_FOLDER + "val"):
        val =datasets.load_from_disk(DATA_FOLDER +"val")

    else:
        print("Creating val emb...")
        val = val.map(preprocess, batched=True)
        print("Saving val emb...")
        val.save_to_disk(DATA_FOLDER +"val")


    if os.path.exists(DATA_FOLDER + "train"):
        train =datasets.load_from_disk(DATA_FOLDER +"train")
    else:
        print("Creating train emb...")
        train = train.map(preprocess, batched=True)
        print("Saving train emb...")
        train.save_to_disk(DATA_FOLDER + "train")

    ################### LOAD MODEL
    #######
    ####

    # loading model config
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
#        finetuning_task="wav2vec2_clf",
        **params
    )

    # load model (with a simple linear projection (input 1024 -> 256 units) and a binary classification on top)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # instantiate a data collator that takes care of correctly padding the input data
    data_collator = DataCollatorCTCWithInputPadding(processor=processor, padding=True)

    if FREEZE_ENCODER and not FREEZE_BASE_MODEL:
        model.freeze_feature_extractor()
    if FREEZE_BASE_MODEL:
        model.freeze_base_model()

    # set arguments to Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True, # can speed up training by batching files of similar length to reduce the amount of padding
        per_device_train_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=10,
        evaluation_strategy="epoch",
        num_train_epochs=EPOCHS,
        log_level="debug",
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=100,
        learning_rate=LEARNING_RATE,  # play with this (also optimizer and learning schedule)
        save_total_limit=2,
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=processor
    )
    # Train!
    print("[INFO] Starting training...")
    trainer.train()
    trainer.evaluate()