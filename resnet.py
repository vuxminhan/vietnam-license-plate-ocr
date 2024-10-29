import torch
from torchvision.transforms import v2 as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import model
import util



class ResNet:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.net = model.Net().cuda() if self.use_cuda else model.Net()
        self.optimizer = None
        self.train_accuracies = []
        self.test_accuracies = []
        self.start_epoch = 1

    def train(self, save_dir, num_epochs=75, batch_size=128, learning_rate=0.001, test_each_epoch=False , verbose=False):
        """Trains the network.

        Parameters
        ----------
        save_dir : str
            The directory in which the parameters will be saved
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        learning_rate : float
            The learning rate
        test_each_epoch : boolean
            True: Test the network after every training epoch, False: no testing
        verbose : boolean
            True: Print training progress to console, False: silent mode
        """
        torch.autograd.set_detect_anomaly(True) 
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.net.train()
        
        train_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomResize(min_size=15, max_size=28),
            transforms.RandomRotation(degrees=(-25, 25), fill=1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=1),
            transforms.ColorJitter(brightness=(0.8, 1.2),
                        contrast=(0.8, 1.2),
                        saturation=(0.8, 1.2),
                        hue=(-0.05, 0.05)),
            transforms.RandomInvert(p=0.15),
            transforms.RandomAffine(
                                    degrees=30,  # Random rotation up to 30 degrees
                                    translate=(0.1, 0.1),  # Random translation of up to 10% of image size
                                    scale=(0.8, 1),  # Random scaling between 80% and 120%
                                    shear=(-10, 10),  # Random shear transformations
                                    fill=1,
                                ),
            transforms.Resize((56, 56), antialias=True),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        train_dataset = ImageFolder(root="./data OCR/train", transform = train_transform)
        num_classes = len(train_dataset.classes)
        print(f"number of classes: {num_classes}")
        print(f"Number of training samples: {len(train_dataset)}")
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")
        data_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        
        criterion = torch.nn.CrossEntropyLoss().cuda() if self.use_cuda else torch.nn.CrossEntropyLoss()

        progress_bar = util.ProgressBar()
        best_accuracy = 0.0
        for epoch in range(self.start_epoch, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))

            epoch_correct = 0
            epoch_total = 0
            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net.forward(images)
                loss = criterion(outputs, labels.squeeze_())
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, dim=1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels.flatten()).sum().item()

                epoch_total += batch_total
                epoch_correct += batch_correct

                if verbose:
                    # Update progress bar in console
                    info_str = 'Last batch accuracy: {:.4f} - Running epoch accuracy {:.4f}'.\
                                format(batch_correct / batch_total, epoch_correct / epoch_total)
                    progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)

                        
            self.train_accuracies.append(epoch_correct / epoch_total)
            if verbose:
                progress_bar.new_line()
                

            if test_each_epoch:
                test_accuracy = self.test()
                self.test_accuracies.append(test_accuracy)
                if verbose:
                    print('Test accuracy: {}'.format(test_accuracy))
                if test_accuracy > best_accuracy:
                    best_accuracy = epoch_correct / epoch_total
                    self.save_parameters(epoch, directory=save_dir)
                    print(f'New best model found at epoch {epoch} with test accuracy {best_accuracy:.4f}. Saving model...')


    def test(self, batch_size=128):
        """Tests the network.

        """
        self.net.eval()

        test_transform = transforms.Compose([   transforms.ToImage(),
                                                transforms.ToDtype(torch.float32, scale=True),
                                                transforms.RandomAffine(
                                                degrees=15,  # Random rotation up to 30 degrees
                                                # translate=(0.1, 0.1),  # Random translation of up to 10% of image size
                                                scale=(0.8, 1),  # Random scaling between 80% and 120%
                                                shear=(-10, 10),  # Random shear transformations
                                                fill=1,
                                            ),
                                                transforms.Resize((56, 56), antialias=True),
                                                transforms.Normalize((0.5,), (0.5,)),
                                             ])

        val_dataset = ImageFolder(root="./dataset/val", transform = test_transform)
        print(f"Number of validation samples: {len(val_dataset)}")

        data_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.flatten()).sum().item()

        self.net.train()
        return correct / total
    
        
    def save_parameters(self, epoch, directory):
        """Saves the parameters of the network to the specified directory.

        Parameters
        ----------
        epoch : int
            The current epoch
        directory : str
            The directory to which the parameters will be saved
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }, \
        os.path.join(directory, 'resnet_transform_augmented.pth'))
        # os.path.join(directory, 'resnet_transform_' + str(epoch) + '.pth'))

    def load_parameters(self, path):
        """Loads the given set of parameters.

        Parameters
        ----------
        path : str
            The file path pointing to the file containing the parameters
        """
        self.optimizer = torch.optim.Adam(self.net.parameters())
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_accuracies = checkpoint['test_accuracies']
        self.start_epoch = checkpoint['epoch']
        print(sum(self.test_accuracies)/len(self.test_accuracies))
