from resnet import ResNet



net = ResNet()
net.load_parameters('saves/resnet_transform_augmented.pth')
net.train(save_dir='saves', num_epochs=201, batch_size=128, learning_rate=0.001, verbose=True, test_each_epoch=True)
accuracy = net.test()
print('Test accuracy: {}'.format(accuracy))