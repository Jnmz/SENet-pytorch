# import packages
import os
import sys

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.utils.data
from net.baseline import resnet20
from net.se_resnet import se_resnet20


def main():
    # Transform configuration and Data Augmentation.
    transform_train = torchvision.transforms.Compose([torchvision.transforms.Pad(4),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.RandomCrop(32),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.5, 0.5, 0.5],
                                                                                       [0.5, 0.5, 0.5])])

    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize([0.5, 0.5, 0.5],
                                                                                      [0.5, 0.5, 0.5])])
    # Load downloaded dataset.
    train_dataset = torchvision.datasets.CIFAR10(root='data', download=True, train=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root='data', download=True, train=False, transform=transform_test)
    # Data Loader.
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    # Make model.
    net_name = 'se-resnet'
    # net_name = 'resnet'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.baseline:
        model = resnet20()
        print('Resnet training')
    else:
        model = se_resnet20(num_classes=args.num_classes, reduction=args.reduction)
        print('SE_Resnet training')
    model.to(device)
    # Loss and optimizer.
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

    resume = True
    if resume:
        if os.path.isfile("senet.pth"):
            print("Resume from checkpoint...")
            checkpoint = torch.load("senet.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initepoch = checkpoint['epoch'] + 2
            print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch'] + 1))
        else:
            print("====>no checkpoint found.")
            initepoch = 1

    writer = SummaryWriter("logs")

    for epoch in range(initepoch - 1, args.epochs):
        # train
        print("-------epoch {} start-------".format(epoch + 1))
        model.train()
        train_acc = 0.0
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss)
            _, predict = torch.max(outputs, dim=1)
            train_acc += torch.eq(predict, labels.to(device)).sum().item()
        train_loss = running_loss / train_steps
        train_accurate = train_acc / train_num
        # val
        model.eval()
        val_acc = 0.0
        running_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for step, val_data in enumerate(val_bar):
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                running_loss += loss.item()
                _, predict = torch.max(outputs, dim=1)
                val_acc += torch.eq(predict, val_labels.to(device)).sum().item()
        val_loss = running_loss / val_steps
        val_accurate = val_acc / val_num
        scheduler.step()
        print('[epoch %d] train_loss: %.3f val_loss:%.3f train_accuracy:%.3f val_accuracy: %.3f' %
              (epoch + 1, train_loss, val_loss, train_accurate, val_accurate))
        writer.add_scalars('loss',
                           {'train': train_loss, 'val': val_loss}, global_step=epoch)
        writer.add_scalars('acc',
                           {'train': train_accurate, 'val': val_accurate}, global_step=epoch)
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch}
        path_checkpoint = "senet.pth"
        torch.save(checkpoint, path_checkpoint)
        print("model saved")
    print('Finished Training')
    writer.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-1)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    main()
