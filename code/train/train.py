import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from dataset import TestDataset, MaskBaseDataset, Sampler
from loss import create_criterion

def seed_everything(seed):
    """
    모든 random seed 고정
    Args:
        param: int
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def increment_path(path, exist_ok=False):
    """
    유효한 path인 지 확인하는 함수
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    Returns: str
    """

    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def get_lr(optimizer):
    """
    optimizer의 현재 학습률을 리턴하는 함수
    Args:
        optimizer: torch.optim

    Returns: float
    """

    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    """
    tensorboard에서 이미지와 실제 label과 예측한 label을 보여주는 함수
    Args:
        np_images: numpy.ndarray
        gts: torch.tensor
        preds: torch.tensor
        n: int
        shuffle: bool

    Returns:
        figure : matplotlib.pyplot.figure
    """

    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    '''
    dataset을 받아 해당 batch size만큼의 idx들로 이루어진 Subset을 반환
    Args:
        dataset: torch.utils.data.Dataset
        train_idx: int
        valid_idx: int
        batch_size: int
        num_workers: int

    Returns:
        train_loader : torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
    '''

    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset, indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset, indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader


def load_model(model_path, num_classes, device):
    """
    model.py에서 작성한 모델을 불러오기
    Args:
        model_path:  f"{args.model_dir}".
        num_classes: int
        device: torch.device

    Returns:
    """

    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def mixup_data(x, y, device):
    """
    Batch 내에서 Random하게 두개의 image를 비율을 두고 섞어
    섞인 하나의 image와 섞이기 전 image들의 class를 반환한다
    Args:
        x: torch.tensor
        y: torch.tensor
        device: torch.device

    Returns:
    """
    lam = np.random.beta(1,1)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, device):
    """
    Batch 내에서 Random하게 두 개의 이미지를 뽑고 image의
    일정 부분을 Crop하여 다른 하나의 이미지 위에 붙인다 붙여
    나온 하나의 image와 이전 두 image들의 class를 반환한다
    Args:
        x: torch.tensor
        y: torch.tensor
        device: torch.device

    Returns:
        x: torch.tensor
        y_a: torch.tensor
        y_b: torch.tensor
        lam: float
    """
    lam = np.random.beta(1,1)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """
    Cutmix 실행 시 Image를 Crop한다
    Args:
        size: torch.tensor
        lam:  float
    Returns:
        bbx1: int
        bby1: int
        bbx2: int
        bby2: int
    """

    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mix_criterion(criterion, pred, y_a, y_b, lam):
    '''
    주어진 lam 비율로 criterion을 사용해 loss를 구한다
    Args:
        criterion: Any
        pred: torch.tensor
        y_a: torch.tensor
        y_b: torch.tensor
        lam: float
    Returns:
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def k_fold(data_dir, model_dir, output_dir, args) :
    '''
    cutmix와 mixup을 통해 data augmentation을 하며 모델을 학습하며
    k개의 모델을 학습해 최종 결과를 도출하는 k-fold ensemble을 진행합니다.
    Args:
        data_dir: String
        model_dir: String
        output_dir: String
        args: argparse.ArgumentParser
    Returns:
    '''
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
    )

    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # meta 데이터와 이미지 경로를 불러옵니다.
    test_img_root = '/opt/ml/input/data/eval/'   
    submission = pd.read_csv(os.path.join(test_img_root, 'info.csv'))
    image_dir = os.path.join(test_img_root, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    test_dataset = TestDataset(image_paths, resize=args.resize)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    skf = StratifiedKFold(n_splits=args.n_splits)

    patience = 5
    accumulation_steps = 2
    mixup_prob = 0.7

    labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]
    
    # K-Fold Cross Validation과 동일하게 Train, Valid Index를 생성합니다. 
    oof_pred = None

    for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
        counter = 0
        best_val_acc = 0
        best_val_loss = np.inf
        train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, args.batch_size, num_workers=4)

        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=os.path.join(save_dir, f'cv{i+1}'))
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            
            for idx, train_batch in enumerate(train_loader):
                inputs, targets = train_batch
                r = np.random.rand(1)

                if r < mixup_prob :
                    if r < mixup_prob/2 :
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets.view(-1,1), device)
                    else :
                        inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets.view(-1,1), device)

                    inputs = inputs.to(device)
                    targets_a = targets_a.squeeze().to(device)
                    targets_b = targets_b.squeeze().to(device)

                    outs = model(inputs)
                    
                    loss = mix_criterion(criterion, outs, targets_a, targets_b, lam)
                else :
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outs = model(inputs)
                    loss = criterion(outs, targets)

                loss.backward()
                
                # -- Gradient Accumulation
                if (idx+1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                loss_value += loss.item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    
                    current_lr = scheduler.get_last_lr()
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    loss_value = 0
                    
            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                val_outputs = []
                val_targets = []
                figure = None
                for val_batch in val_loader:
                    inputs, targets = val_batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, targets).item()
                    acc_item = (targets == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    val_outputs.append(preds)
                    val_targets.append(targets)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, targets, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )
                val_outputs = torch.cat(val_outputs, 0)
                val_targets = torch.cat(val_targets, 0)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_outputs_np = torch.clone(val_outputs).detach().cpu().numpy()
                val_targets_np = torch.clone(val_targets).detach().cpu().numpy()
                val_acc = f1_score(val_outputs_np, val_targets_np, average='macro')

                # Callback1: validation accuracy가 향상될수록 모델을 저장합니다.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if val_acc > best_val_acc:
                    print("New best model for val accuracy! saving the model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/cv{i+1}.pth")
                    best_val_acc = val_acc
                    counter = 0
                else:
                    counter += 1
                # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
                if counter > patience:
                    print("Early Stopping...")
                    break

                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()
                
        # 각 fold에서 생성된 모델을 사용해 Test 데이터를 예측합니다. 
        model = load_model(f"{save_dir}/cv{i+1}.pth", num_classes, device).to(device)
        all_predictions = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)

                # Test Time Augmentation
                pred = model(images) / 2 # 원본 이미지를 예측하고
                pred += model(torch.flip(images, dims=(-1,))) / 2 # horizontal_flip으로 뒤집어 예측합니다. 
                all_predictions.extend(pred.cpu().numpy())

            fold_pred = np.array(all_predictions)

        # 확률 값으로 앙상블을 진행하기 때문에 'k'개로 나누어줍니다.
        if oof_pred is None:
            oof_pred = fold_pred / args.n_splits
        else:
            oof_pred += fold_pred / args.n_splits

    submission['ans'] = np.argmax(oof_pred, axis=1)
    submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)

    print('test inference is done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--n_splits', type=int, default=5, help='k for stratified k-fold')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output/kfold'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    k_fold(data_dir, model_dir, output_dir, args)