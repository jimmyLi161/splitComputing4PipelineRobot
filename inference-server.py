import os
import time
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import model_alpha.bn_models
from dataloader import MultiLabelDatasetInference


def evaluate(dataloader, model, device):
    model.eval()

    sigmoidPredictions = None
    imgPathsList = []

    sigmoid = nn.Sigmoid()

    dataLen = len(dataloader)
    eljx = 0
    with torch.no_grad():
        for i, (images, imgPaths) in enumerate(dataloader):
            if i % 100 == 0:
                print("{} / {}".format(i, dataLen))

            images = images.to(device)
            print("1")

            output = model.bottleneck.encoder(images)
            print("2")
            # a = output.cpu()
            # np.save("{}_opm.npy".format(eljx),a)
            # eljx = eljx + 1

            b = np.load("D:\\TJU\\test\\Nt\\npy\\{}_opm.npy".format(eljx))
            print("3")
            eljx = eljx + 1
            opm = torch.tensor(b).cuda()

            output = model.bottleneck.decoder(output)
            output = model.layer3(output)
            output = model.layer4(output)
            output = model.avgpool(output)
            output = model.flatten(output)
            output = model.fc(output)

            sigmoidOutput = sigmoid(output).detach().cpu().numpy()

            if sigmoidPredictions is None:
                sigmoidPredictions = sigmoidOutput
            else:
                sigmoidPredictions = np.vstack((sigmoidPredictions, sigmoidOutput))

            imgPathsList.extend(list(imgPaths))
    return sigmoidPredictions, imgPathsList


def load_model(model_path, best_weights=False):

    if best_weights:
        # print("1")
        if not os.path.isfile(model_path):
            raise ValueError("The provided path does not lead to a valid file: {}".format(model_path))
        last_ckpt_path = model_path
    else:
        # print("2")
        last_ckpt_path = os.path.join(model_path, "last.ckpt")
        # print(last_ckpt_path)
        if not os.path.isfile(last_ckpt_path):
            raise ValueError("The provided directory path does not contain a 'last.ckpt' file: {}".format(model_path))
    
    model_last_ckpt = torch.load(last_ckpt_path)
    # print(last_ckpt_path)

    model_name = model_last_ckpt["hyper_parameters"]["model"]
    num_classes = model_last_ckpt["hyper_parameters"]["num_classes"]
    training_mode = model_last_ckpt["hyper_parameters"]["training_mode"]
    br_defect = model_last_ckpt["hyper_parameters"]["br_defect"]
    
    # Load best checkpoint

    if best_weights:
        best_model = model_last_ckpt
    else:
        best_model_path = model_last_ckpt["checkpoint_callback_best_model_path"]
        # print(best_model_path)
        # best_model_path = './log/r101-3/e2e-version_1/epoch=55-val_loss=1.06.ckpt'
        best_model = torch.load(best_model_path)

    best_model_state_dict = best_model["state_dict"]

    updated_state_dict = OrderedDict()
    for k,v in best_model_state_dict.items():
        name = k.replace("model.", "")
        if "criterion" in name:
            continue

        updated_state_dict[name] = v

    return updated_state_dict, model_name, num_classes, training_mode, br_defect


def run_inference(args):

    ann_root = args["ann_root"]
    data_root = args["data_root"]
    model_path = args["model_path"]
    outputPath = args["results_output"]
    # best_weights = args["best_weights"]
    split = args["split"]

    best_weights = False
    
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
  
    updated_state_dict, model_name, num_classes, training_mode, br_defect = load_model(model_path, best_weights)

    if "model_version" not in args.keys():
        model_version = model_name
    else:
        model_version = args["model_version"]

    model = model_alpha.bn_models.SplitResNet101(bottleneck_channel=3, bottleneck_idx=7, compressor=None, decompressor=None,
                     short_module_names=None, num_classes=num_classes)
    model.load_state_dict(updated_state_dict)
    # print(model.bottleneck.encoder)
    # initialize dataloaders
    img_size = 299 if model in ["inception_v3", "chen2018_multilabel"] else 224
    
    eval_transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
        ])

        
    dataset = MultiLabelDatasetInference(ann_root, data_root, split=split, transform=eval_transform, onlyDefects=False)
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], num_workers = args["workers"], pin_memory=True)

    if training_mode in ["e2e", "defect"]:
        labelNames = dataset.LabelNames
    elif training_mode == "binary":
        labelNames = ["Defect"]
    elif training_mode == "binaryrelevance":
        labelNames = [br_defect]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Validation results
    print("VALIDATION")
    sigmoid_predictions, val_imgPaths = evaluate(dataloader, model, device)
    print("A")
    sigmoid_dict = {}
    print("B")    
    sigmoid_dict["Filename"] = val_imgPaths
    print("C")
    for idx, header in enumerate(labelNames):
        sigmoid_dict[header] = sigmoid_predictions[:,idx]
    print("D")
    sigmoid_df = pd.DataFrame(sigmoid_dict)
    print("E")
    sigmoid_df.to_csv(os.path.join(outputPath, "{}_{}_sigmoid.csv".format(model_version, split.lower())), sep=",", index=False)
    print("F")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations')
    parser.add_argument('--data_root', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=512, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--best_weights", action="store_true", help="If true 'model_path' leads to a specific weight file. If False it leads to the output folder of lightning_trainer where the last.ckpt file is used to read the best model weights.")
    parser.add_argument("--results_output", type=str, default = "./results")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])

    args = vars(parser.parse_args())

    run_inference(args)