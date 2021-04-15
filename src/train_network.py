import pathlib
import torch
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from defines import *
from model.losses import focal, smooth_L1
from model.architectures.vgg import VGG_Retinanet
from data_preprocessing.data_generator import dataloader


if __name__ == "__main__":
    writer = SummaryWriter('runs/CRFNET_1')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VGG_Retinanet(num_classes=8, backbone=cfg.network, weights=None, include_top=False, cfg=cfg)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        classification_loss = 0
        regression_loss = 0
        for idx, data in enumerate(dataloader, 0):
            inputs, targets = data
            targets = [t.to(device=device) for t in targets]
            inputs = inputs.to(device=device)
            # print(torch.cuda.memory_summary(device=device, abbreviated=False))

            optimizer.zero_grad()

            outputs = model(inputs)

            L1_loss = smooth_L1(y_pred=outputs[0], y_true=targets[0])
            focal_loss = focal(y_pred=outputs[1], y_true=targets[1])
            focal_loss.backward(retain_graph=True), L1_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            # print statistics
            classification_loss += focal_loss.item()
            regression_loss += L1_loss.item()
            if idx % 100 == 99:  # print every 2000 mini-batches
                print(f"[{epoch+1}, {idx+1}] classification_loss = {classification_loss/100}, "
                      f"regression_loss = {regression_loss/100}")
                # ...log the running loss
                writer.add_scalar("Regression Loss", regression_loss / 100)
                writer.add_scalar("Classification Loss", classification_loss / 100)
                classification_loss = 0
                regression_loss = 0
        if cfg.save_model:
            PATH = pathlib.Path.joinpath(cfg.save_model, f"model1_epoch{epoch+1}")
            torch.save(model.state_dict(), PATH)
            print(f"Saved Model at Epoch {epoch+1} at {PATH}")

    print('Finished Training')
    writer.flush()
    writer.close()



