import argparse
parser = argparse.ArgumentParser()

# Model
parser.add_argument("--model", help="The name of used model", default='ResUnetPlusPlus')

# Dataset
parser.add_argument("--checkpoint-path", help="Check point directory", default='./checkpoints/')
parser.add_argument("--pretrained-path", help="Saved model directory", default='./saved_models/')
parser.add_argument("--infer-path", help="Inference directory", default='./predicted_masks/')
parser.add_argument("--images-path", help="Image directory", default='./data/train/train/')
parser.add_argument("--masks-path", help="Masks directory", default='./data/train_gt/train_gt/')
parser.add_argument("--test-path", help="Test directory", default='./data/test/test/')

# Training
parser.add_argument("--num-epochs", help="Number of training epochs", type=int, default=2)
parser.add_argument("--lr", help="Initial learning rate", type=float, default=2e-4)
parser.add_argument("--batch-size", help="Batch size", type=int, default=4)
parser.add_argument("--display-step", help="Log the training process every n steps", type=int, default=50)
parser.add_argument("--train-size", help="Size of training dataset (propotional to the whole dataset)", default=0.8)
parser.add_argument("--valid-size", help="Size of valid dataset (propotional to the whole dataset)", default=0.2)
parser.add_argument("--weights", help="Load weights into model", default='')
