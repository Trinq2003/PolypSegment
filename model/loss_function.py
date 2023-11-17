import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeometry.losses import one_hot

class CEDiceLoss(nn.Module):
    def __init__(self, weights) -> None:
        super(CEDiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.weights: torch.Tensor = weights

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, target.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        if not self.weights.shape[1] == input.shape[1]:
            raise ValueError("The number of weights must equal the number of classes")
        if not torch.sum(self.weights).item() == 1:
            raise ValueError("The sum of all weights must equal 1")

        # cross entropy loss
        celoss = nn.CrossEntropyLoss(self.weights)(input, target)
        
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)
        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        
        dice_score = torch.sum(dice_score * self.weights, dim=1)
        
        return torch.mean(1. - dice_score) + celoss

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        if not self.weights.shape[1] == input.shape[1]:
            raise ValueError("The number of weights must equal the number of classes")
        if not torch.sum(self.weights).item() == 1:
            raise ValueError("The sum of all weights must equal 1")

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)

# class BCEDiceLoss(nn.Module):
#     def __init__(self, weights=None) -> None:
#         super(BCEDiceLoss, self).__init__()
#         self.eps: float = 1e-6
#         self.weights: torch.Tensor = weights

#     def forward(
#             self,
#             input: torch.Tensor,
#             target: torch.Tensor) -> torch.Tensor:
#         if not torch.is_tensor(input):
#             raise TypeError("Input type is not a torch.Tensor. Got {}"
#                             .format(type(input)))
#         if not len(input.shape) == 4:
#             raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
#                              .format(input.shape))
#         if not input.shape[-2:] == target.shape[-2:]:
#             raise ValueError("input and target shapes must be the same. Got: {}"
#                              .format(input.shape, target.shape))
#         if not input.device == target.device:
#             raise ValueError(
#                 "input and target must be in the same device. Got: {}" .format(
#                     input.device, target.device))
#         if self.weights is not None and not self.weights.shape[0] == input.shape[1]:
#             raise ValueError("The number of weights must equal the number of classes")
        
#         target_one_hot = one_hot(target, num_classes=input.shape[1],
#                                  device=input.device, dtype=input.dtype)
        
#         # compute sigmoid over the classes axis
#         input_sigmoid = torch.sigmoid(input)


#         # compute the actual dice score
#         dims = (2, 3)
#         intersection = torch.sum(input_sigmoid * target_one_hot, dims)
#         cardinality = torch.sum(input_sigmoid + target_one_hot, dims)

#         dice_score = 2. * intersection / (cardinality + self.eps)
#         print(f"[INFO] Dice score: {dice_score}")
        
#         if self.weights is not None:
#             dice_score = torch.sum(dice_score * self.weights, dim=1)
#         else:
#             dice_score = torch.mean(dice_score, dim=1)
        
#         return torch.mean(bce_loss) + (1. - torch.mean(dice_score))

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BCEDiceLoss(nn.Module):
#     def __init__(self, weights) -> None:
#         super(BCEDiceLoss, self).__init__()
#         self.eps: float = 1e-6
#         self.weights: torch.Tensor = weights

#     def forward(
#             self,
#             input: torch.Tensor,
#             target: torch.Tensor) -> torch.Tensor:
#         if not torch.is_tensor(input):
#             raise TypeError("Input type is not a torch.Tensor. Got {}"
#                             .format(type(input)))
#         if not len(input.shape) == 4:
#             raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
#                              .format(input.shape))
#         if not input.shape[-2:] == target.shape[-2:]:
#             raise ValueError("input and target shapes must be the same. Got: {}"
#                              .format(input.shape, target.shape))
#         if not input.device == target.device:
#             raise ValueError(
#                 "input and target must be on the same device. Got: {}" .format(
#                     input.device, target.device))
#         if not self.weights.shape[1] == input.shape[1]:
#             raise ValueError("The number of weights must equal the number of classes")
#         if not torch.sum(self.weights).item() == 1:
#             raise ValueError("The sum of all weights must equal 1")

#         # compute softmax over the classes axis
#         input_soft = F.softmax(input, dim=1)

#         # create the labels one hot tensor
#         target_one_hot = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()
#         dims = (2, 3)
#         intersection = torch.sum(input_soft[:, 1:, :, :] * target_one_hot[:, 1:, :, :], dims)
#         cardinality = torch.sum(input_soft[:, 1:, :, :] + target_one_hot[:, 1:, :, :], dims)

#         dice_score = 2. * intersection / (cardinality + self.eps)

#         dice_score = torch.sum(dice_score * self.weights[:, 1:], dim=1)

#         # binary cross entropy loss
#         bceloss = nn.BCEWithLogitsLoss()(input[:, 0, :, :], target.float())

#         return torch.mean(1. - dice_score) + bceloss