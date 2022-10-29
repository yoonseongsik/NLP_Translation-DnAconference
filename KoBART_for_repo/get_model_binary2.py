import argparse
from train import KoBARTConditionalGeneration
from transformers.models.bart import BartForConditionalGeneration
import yaml



parser = argparse.ArgumentParser()

parser.add_argument("--hparams", default=None, type=str)
parser.add_argument("--model_binary", default=None, type=str)
parser.add_argument("--output_dir", default='kobart_translation', type=str)

args = parser.parse_args('')
args.hparams = '/content/drive/Shareddrives/D&A/모델/KoBART/KoBART-translation-main/logs/tb_logs/default/version_9/hparams.yaml'                                                         ## 경로지정 요망
args.model_binary = '/content/drive/Shareddrives/D&A/모델/KoBART/KoBART-translation-main/logs/kobart_translation-model_chp/epoch=02-val_loss=0.628.ckpt'                                  ## 경로지정 요망

with open(args.hparams) as f:
    hparams = yaml.load(f)
    
inf = KoBARTConditionalGeneration.load_from_checkpoint(args.model_binary, hparams=hparams)

inf.model.save_pretrained(args.output_dir)

