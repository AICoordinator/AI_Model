import torchvision
from torch import nn
import torch
from dataset import ImageDataset
from tensorboardX import SummaryWriter
from torchvision import transforms
from networks import RegressionNetwork
import os

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="test")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument("--dataroot", default="/home/nas1_userB/dataset/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs_zalando.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--mtviton_checkpoint', type=str, default='', help='mtviton checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=300000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument('--Ddownx2', action='store_true', help="Downsample D's input to increase the receptive field")  
    parser.add_argument('--Ddropout', action='store_true', help="Apply dropout to D")
    parser.add_argument('--num_D', type=int, default=2, help='Generator ngf')
    # training
    parser.add_argument("--G_D_seperate", action='store_true')
    parser.add_argument("--no_GAN_loss", action='store_true')
    parser.add_argument("--lasttvonly", action='store_true')
    parser.add_argument("--interflowloss", action='store_true', help="Intermediate flow loss")
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
    parser.add_argument('--edgeawaretv', type=str, choices=['no_edge', 'last_only', 'weighted'], default="no_edge", help="Edge aware TV loss")
    parser.add_argument('--add_lasttv', action='store_true')
    
    # test visualize
    parser.add_argument("--no_test_visualize", action='store_true')    
    parser.add_argument("--num_test_visualize", type=int, default=3)
    parser.add_argument("--test_datasetting", default="unpaired")
    parser.add_argument("--test_dataroot", default="/home/nas2_userF/gyojunggu/WUTON/data/zalando-hd-resize")
    parser.add_argument("--test_data_list", default="test_pairs.txt")
    

    # Hyper-parameters
    parser.add_argument('--G_lr', type=float, default=0.0002, help='Generator initial learning rate for adam')
    parser.add_argument('--D_lr', type=float, default=0.0002, help='Discriminator initial learning rate for adam')
    parser.add_argument('--CElamda', type=float, default=10, help='initial learning rate for adam')
    parser.add_argument('--GANlambda', type=float, default=1)
    parser.add_argument('--tvlambda', type=float, default=2)
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--val_count', type=int, default='1000')
    parser.add_argument('--spectral', action='store_true', help="Apply spectral normalization to D")
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")
    
    opt = parser.parse_args()
    return opt

def visualize(test_iter, model):
    image, label = test_iter.next()

def train(model, train_loader, test_loader, opt):
    model.train()
    model.cuda()

    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    # Define loss function as MSE Loss
    criterion = nn.MSELoss()
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Train the model
    iter = 0
    # Train Loop
    for iter in range(opt.max_iter):
        # Get the inputs
        try:
            image, label = train_iter.next()
        except:
            train_iter = iter(train_loader)
            image, label = train_iter.next()
        image = image.cuda()
        label = label.cuda()
        # Forward pass
        outputs = model(image)
        loss = criterion(outputs, label)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Log the loss
        if iter % 100 == 0:
            print('Iter: {}\tLoss: {:.6f}'.format(iter, loss.item()))
            board.add_scalar('Loss/train', loss.item(), iter + 1)

        # Validation
        if iter % opt.val_count == 0:
            model.eval()
            with torch.no_grad():
                val_loss_list = []
                for i, (image, label) in enumerate(test_loader):
                    image = image.cuda()
                    label = label.cuda()
                    outputs = model(image)
                
                    val_loss = criterion(outputs, label)

                    if i == 0:
                        board.add_image('Validation/image0', image[0], iter + 1)
                        board.add_image('Validation/image1', image[1], iter + 1)
                        board.add_image('Validation/image2', image[2], iter + 1)
                        board.add_image('Validation/image3', image[3], iter + 1)
                        board.add_image('Validation/image4', image[4], iter + 1)

                        board.add_scalar('Validation/label0', label[0], iter + 1)
                        board.add_scalar('Validation/label1', label[1], iter + 1)
                        board.add_scalar('Validation/label2', label[2], iter + 1)
                        board.add_scalar('Validation/label3', label[3], iter + 1)
                        board.add_scalar('Validation/label4', label[4], iter + 1)
                        
                    val_loss_list.append(val_loss.item())
                val_loss = sum(val_loss_list)/len(val_loss_list)
                print('Validation loss: {:.6f}'.format(val_loss))
                board.add_scalar('Loss/val', val_loss, iter + 1)
                # Pick 3 images and predicted values from validation set for visualization
                test_iter = iter(test_loader)
                

            model.train()

if __name__ == '__main__':
    opt = get_opt()
    # Define model
    model = RegressionNetwork()
    # Define dataloader
    train_dataset = ImageDataset(data_dir=opt.dataroot, label_dir=opt.label_dir)
    test_dataset = ImageDataset(data_dir=opt.test_dataroot, label_dir=opt.test_label_dir)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
    # Train the model
    train(model, train_loader, test_loader, opt)
