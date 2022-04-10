import torchvision
from torch import nn
import torch
from dataset import ImageDataset
from tensorboardX import SummaryWriter
from torchvision import transforms
from networks import ResNet18, resnext50_32x4d
import argparse
from torchvision.utils import save_image
import os

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="test")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument("--dataroot", default="/home/sangyunlee/dataset/SCUT-FBP5500_v2/Images")
    parser.add_argument("--label_dir", default="/home/sangyunlee/dataset/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/train.txt")
    parser.add_argument("--mask_dir", default="/home/sangyunlee/dataset/SCUT-FBP5500_v2/mask")
    parser.add_argument("--test_dataroot", default="/home/sangyunlee/dataset/SCUT-FBP5500_v2/Images")
    parser.add_argument("--test_label_dir", default="/home/sangyunlee/dataset/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/test.txt")

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default="/home/sangyunlee/AI_Model/attractiveness/checkpoints/resnet18-5c106cde.pth")
    parser.add_argument("--max_iter", type=int, default=400000)
    parser.add_argument("--val_count", type=int, default=1000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
  
    


    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='Generator initial learning rate for adam')
    
    
    opt = parser.parse_args()
    return opt

def visualize(test_iter, model):
    image, label = test_iter.next()

def train(model, train_loader, test_loader, opt):
    model.train()
    model.cuda()

    # Check path
    if not os.path.exists(os.path.join(opt.checkpoint_dir, opt.name)):
        os.mkdir(os.path.join(opt.checkpoint_dir, opt.name))
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    # Define loss function as MSE Loss
    criterion = nn.MSELoss()
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
            max(0, step * 1000 - opt.max_iter // 2) / float(opt.max_iter // 2 + 1))
    # Train the model
    # Train Loop
    best_val_loss = float('inf')
    for step in range(opt.max_iter):
        # Get the inputs
        try:
            image, label, mask, _ = train_iter.next()
        except:
            train_iter = iter(train_loader)
            image, label, mask, _ = train_iter.next()
        image = image.cuda()
        label = label.cuda()
        mask = mask.cuda()
        # Forward pass
        outputs = model(image, mask)
        assert outputs.shape == label.shape, f"outputs.shape: {outputs.shape}, label.shape: {label.shape}"
        loss = criterion(outputs, label)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Log the loss
        if step % 100 == 0:
            print('Step: {}\tLoss: {:.6f}'.format(step, loss.item()))
            board.add_scalar('Loss/train', loss.item(), step + 1)

        # Validation
        if step % opt.val_count == 0:
            model.eval()
            with torch.no_grad():
                val_loss_list = []
                for i, (image, label, mask, _) in enumerate(test_loader):
                    image = image.cuda()
                    label = label.cuda()
                    mask = mask.cuda()
                    outputs = model(image, mask)
                
                    val_loss = criterion(outputs, label)

                    if i == 0:
                        board.add_image('Validation/image0', image[0] * 0.5 + 0.5, step + 1)
                        board.add_image('Validation/image1', image[1] * 0.5 + 0.5, step + 1)
                        board.add_image('Validation/image2', image[2] * 0.5 + 0.5, step + 1)
                        board.add_image('Validation/image3', image[3] * 0.5 + 0.5, step + 1)
                        board.add_image('Validation/image4', image[4] * 0.5 + 0.5, step + 1)

                        # print("mask.shape: ", mask.shape, "image.shape: ", image.shape)
                        board.add_images('Validation/mask', mask[0][0], step + 1, dataformats='HW')

                        board.add_scalar('Validation/label0', label[0], step + 1)
                        board.add_scalar('Validation/label1', label[1], step + 1)
                        board.add_scalar('Validation/label2', label[2], step + 1)
                        board.add_scalar('Validation/label3', label[3], step + 1)
                        board.add_scalar('Validation/label4', label[4], step + 1)

                        board.add_scalar('Validation/output0', outputs[0], step + 1)
                        board.add_scalar('Validation/output1', outputs[1], step + 1)
                        board.add_scalar('Validation/output2', outputs[2], step + 1)
                        board.add_scalar('Validation/output3', outputs[3], step + 1)
                        board.add_scalar('Validation/output4', outputs[4], step + 1)  

                    val_loss_list.append(val_loss.item())
                val_loss = sum(val_loss_list)/len(val_loss_list)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(opt.checkpoint_dir, opt.name, 'best_model.pth'))
                
                print(f"{step}Validation Loss: {val_loss}, Best Validation Loss: {best_val_loss}")
                board.add_scalar('Loss/val', val_loss, step + 1)
                board.add_scalar('Loss/best_val', best_val_loss, step + 1)



                
                

            model.train()
            # Save the model
        if step % opt.save_count == 0:
            torch.save(model.state_dict(), os.path.join(opt.checkpoint_dir, opt.name, f'{step}.pth'))
        if step % 1000 == 0:
            scheduler.step()

if __name__ == '__main__':
    opt = get_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    # Define model
    # model = ResNet18()
    model = resnext50_32x4d()
    # Load checkpoint
    if opt.ckpt:
        model.load_state_dict(torch.load(opt.ckpt), strict=False)
        
    # Define dataloader
    transform_train = [  # transforms.RandomRotation(degrees=(0, 180)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Resize((256, 256)), 
                                            transforms.RandomCrop(224), 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                                                                  ]
    transform_test = [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),]
    train_dataset = ImageDataset(data_dir=opt.dataroot, label_dir=opt.label_dir, mask_dir = opt.mask_dir, transform=transform_train)
    test_dataset = ImageDataset(data_dir=opt.test_dataroot, label_dir=opt.test_label_dir, mask_dir = opt.mask_dir, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
    # Train the model
    train(model, train_loader, test_loader, opt)
