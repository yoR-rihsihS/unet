import os
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from torch.amp import GradScaler

from unet import UNet, FocalLoss, train_one_epoch, evaluate
from cityscapes import CityScapes, get_transforms

DEVICE = 'cuda'
torch.cuda.empty_cache()


def save_file(history, path):
    with open(path, 'wb') as file:
        pickle.dump(history, file)

def load_file(path):
    with open(path, 'rb') as file:
        history = pickle.load(file)
    return history

def print_metrics(train_metrics, val_metrics, epoch):
    print(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f} - Val Loss: {val_metrics['loss']:.4f}")
    print(f"Train IOU: {train_metrics['mean_iou']:.4f} - Val IOU: {val_metrics['mean_iou']:.4f}")
    print(f"Train Dice: {train_metrics['mean_dice']:.4f} - Val Dice: {val_metrics['mean_dice']:.4f}")
    print(f"Train Pixel Accuracy: {train_metrics['mean_pixel_accuracy']:.4f} - Val Pixel Accuracy: {val_metrics['mean_pixel_accuracy']:.4f}")
    print()

def main(cfg):
    transform_train, transform_val_test = get_transforms(cfg["train_crop_size"], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    train_set = CityScapes(root_dir='./data/', split='train', transform=transform_train)
    val_set = CityScapes(root_dir='./data/', split='val', transform=transform_val_test)

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, persistent_workers=True, prefetch_factor=10, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"] // 4, shuffle=False, num_workers=2, persistent_workers=True, prefetch_factor=10, pin_memory=True)

    model = UNet(num_classes=cfg['num_classes'])
    model.to(DEVICE)

    print("Model summary:")
    summary(model, input_size=(1, 3, 1024, 2048), device=DEVICE)

    num_parameters = sum(p.numel() for p in model.parameters())
    print("Number of parameters =", num_parameters)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters =", num_parameters)

    optimizer = optim.AdamW(model.parameters(), weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step"], gamma=cfg['gamma'])
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0, size_average=True, ignore_index=cfg["ignore_class"])
    scaler = GradScaler()

    history = {"train": [], "val": []}

    if os.path.exists(f"./saved/unet_checkpoint.pth"):
        history = load_file(f"./saved/unet_history.pkl")
        checkpoint = torch.load(f"./saved/unet_checkpoint.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    for epoch in range(cfg["epochs"]):
        if len(history['train']) > epoch:
            print_metrics(history['train'][epoch], history['val'][epoch], epoch+1)
            print()
            continue

        train_loss, train_iou_per_class, train_dice_per_class, train_pixel_accuracy_per_class = train_one_epoch(model, train_loader, loss_fn, optimizer, cfg["num_classes"], scaler, DEVICE)
        val_loss, val_iou_per_class, val_dice_per_class, val_pixel_accuracy_per_class = evaluate(model, val_loader, loss_fn, cfg["num_classes"], DEVICE)
        ignore_index_mask = torch.arange(cfg["num_classes"]).to(DEVICE) != cfg["ignore_class"]

        scheduler.step()

        history['train'].append({
            'loss': train_loss,
            'mean_iou': train_iou_per_class[ignore_index_mask].mean().cpu().item(),
            'mean_dice': train_dice_per_class[ignore_index_mask].mean().cpu().item(),
            'mean_pixel_accuracy': train_pixel_accuracy_per_class[ignore_index_mask].mean().cpu().item(),
        })

        history['val'].append({
            'loss': val_loss,
            'mean_iou': val_iou_per_class[ignore_index_mask].mean().cpu().item(),
            'mean_dice': val_dice_per_class[ignore_index_mask].mean().cpu().item(),
            'mean_pixel_accuracy': val_pixel_accuracy_per_class[ignore_index_mask].mean().cpu().item(),
        })

        print_metrics(history['train'][epoch], history['val'][epoch], epoch+1)

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, f"./saved/unet_checkpoint.pth")
        save_file(history, f"./saved/unet_history.pkl")

    torch.save(model.state_dict(), f"./saved/unet_e_{epoch+1}.pth")

if __name__=="__main__":
    config = {
        'batch_size': 6,
        'ignore_class': 19,
        'train_crop_size': [1024, 1024],
        'num_classes': 20,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'step': 10,
        'gamma': 0.1,
    }
    main(config)