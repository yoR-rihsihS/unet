import os
import argparse
import numpy as np
from time import time
from PIL import Image

import torch

from unet import UNet, convert_trainid_mask
from cityscapes import get_transforms

DEVICE = 'cuda'
torch.cuda.empty_cache()

def inference(model, data_path, save_path, transform):
    time_taken = []
    model.eval()
    for city in os.listdir(data_path):
        img_dir = os.path.join(data_path, city)
        tgt_dir = os.path.join(save_path, city)
        os.makedirs(tgt_dir, exist_ok=True)
        for file_name in os.listdir(img_dir):
            if file_name.endswith('_leftImg8bit.png'):
                img_path = os.path.join(img_dir, file_name)
                tgt_name = file_name.replace('_leftImg8bit.png', '_labelId_preds.png')
                tgt_path = os.path.join(tgt_dir, tgt_name)

                start = time()

                image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
                pred_logits = model(image)
                pred_mask = torch.argmax(pred_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                pred_mask_color = convert_trainid_mask(
                    pred_mask,
                    to="labelid",
                    name_to_trainId_path='./cityscapes/name_to_trainId.json',
                    name_to_color_path='./cityscapes/name_to_color.json',
                    name_to_labelId_path='./cityscapes/name_to_labelId.json',
                ).astype(np.uint8)
            
                end = time()

                img = Image.fromarray(pred_mask_color, mode='L')
                img.save(tgt_path)
                time_taken.append(end - start)

    print(f"Model inference on images from {data_path} has been recorded in {save_path}")
    print(f"Average time to run model on one image : {(sum(time_taken) / len(time_taken)):.4f}")


def main(cfg):
    transform_train, transform_val_test = get_transforms(cfg["train_crop_size"], cfg["norm_mean"], cfg["norm_std"])

    model = UNet(num_classes=cfg['num_classes'])
    model_state_dict = torch.load(cfg['model_weights_path'], map_location='cpu', weights_only=True)
    model.load_state_dict(model_state_dict)
    model.to(DEVICE)
    
    inference(model, data_path='./data/leftImg8bit/test/', save_path=f"./outputs/unet/", transform=transform_val_test)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="UNet Training")
    parser.add_argument("--model_weights_path", type=str, required=True, help="Path to the model weights")
    args = parser.parse_args()
    model_weights_path = args.model_weights_path
    config = {
        'ignore_class': 19,
        'train_crop_size': [1024, 1024],
        'norm_mean': [0.0, 0.0, 0.0],
        'norm_std': [1.0, 1.0, 1.0],
        'num_classes': 20,
        'model_weights_path': model_weights_path,
    }
    main(config)