import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

def weights_init(model):
    if isinstance(model, nn.Linear):
        # Xavier Distribution
        torch.nn.init.xavier_uniform_(model.weight)

def save_model(model, optimizer, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def learning_curve_plotting(epochs, train_loss_array):
    plt.rcParams['figure.dpi'] = 90
    plt.rcParams['figure.figsize'] = (6, 4)
    epochs_array = range(epochs)
    # Plot Training and Test loss
    plt.plot(epochs_array, train_loss_array, 'g', label='Training loss')
    # plt.plot(epochs_array, test_loss_array, 'b', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def result_visualization(model, train_dataloader):
    for i, (data, label) in enumerate(train_dataloader):
        img = data
        mask = label
        break
    fig, arr = plt.subplots(4, 3, figsize=(16, 12))
    arr[0][0].set_title('Image')
    arr[0][1].set_title('Segmentation')
    arr[0][2].set_title('Predict')

    model.eval()
    with torch.no_grad():
        predict = model(img)

    for i in range(4):
        arr[i][0].imshow(img[i].permute(1, 2, 0));
        
        arr[i][1].imshow(F.one_hot(mask[i]).float())
        
        arr[i][2].imshow(F.one_hot(torch.argmax(predict[i], 0).cpu()).float())