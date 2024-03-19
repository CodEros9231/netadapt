# import torch
# import torch.nn as nn
# from torchvision import models

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
    
# #ADDED LATER ON FOR Resnet backboning

# class ASPPModule(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation):
#         super(ASPPModule, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return self.relu(x)

# class DeepLabV3Plus(nn.Module):
#     def __init__(self):
#         super(DeepLabV3Plus, self).__init__()
#         # Load a pretrained ResNet-50 model
#         self.resnet50 = models.resnet50(pretrained=True)
#         # Remove the fully connected layers
#         self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])

#         # ASPP Phase
#         self.aspp = ASPPModule(in_channels=2048)  # ResNet50 feature size at block 4

#         # LLF Phase
#         self.llf_conv = ConvBlock(in_channels=512, out_channels=48, kernel_size=1, padding=0, dilation=1)  # ResNet50 feature size at block 2

#         # Top convolutional layers after concatenation
#         self.top_conv1 = ConvBlock(in_channels=256+48, out_channels=256, kernel_size=3, padding=1, dilation=1)
#         self.top_conv2 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, dilation=1)

#         # Output layer
#         self.output_conv = nn.Conv2d(256, 3, kernel_size=1, padding=0)  # 3 for number of classes, adjust as needed

#     def forward(self, x):
#         # Pass input through ResNet50
#         resnet_output = self.resnet50(x)

#         # ASPP Phase
#         aspp_output = self.aspp(resnet_output)
#         aspp_output = nn.functional.interpolate(aspp_output, size=x.shape[2:], mode='bilinear', align_corners=False)

#         # LLF Phase
#         llf_output = self.resnet50[:5](x)  # Output of block 2 in ResNet50
#         llf_output = self.llf_conv(llf_output)

#         # Combine ASPP and LLF
#         combined = torch.cat([aspp_output, llf_output], dim=1)

#         # Top Convolutional Layers
#         top_output = self.top_conv1(combined)
#         top_output = self.top_conv2(top_output)

#         # Output Layer
#         pred_mask = self.output_conv(top_output)

#         return pred_mask

# # Create the DeepLabV3+ model
# model = DeepLabV3Plus()
# print(model)

# # added portion for resnet backboning ends

# class AtrousSpatialPyramidPooling(nn.Module):
#     def __init__(self, in_channels,num_classes=10):
#         super(AtrousSpatialPyramidPooling, self).__init__()
#         out_channels = in_channels // 4
#         print("in_channels:", in_channels, "out_channels:", out_channels)

#         self.global_avg_pool = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True))
#         # Print the shape of the weights for debugging
#         print("global_avg_pool weights:", self.global_avg_pool[1].weight.shape)

#         self.aspp1 = ASPPModule(in_channels, out_channels, dilation=1)
#         self.aspp2 = ASPPModule(in_channels, out_channels, dilation=6)
#         self.aspp3 = ASPPModule(in_channels, out_channels, dilation=12)
#         self.aspp4 = ASPPModule(in_channels, out_channels, dilation=18)
    
#         self.concat_projection = nn.Sequential(
#             nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5))
        
#         print("concat_projection weights:", self.concat_projection[0].weight.shape)


#     def forward(self, x):
#         print("Input shape:", x.shape)

#         x1 = self.global_avg_pool(x)
#         x1 = nn.functional.interpolate(x1, size=x.shape[2:], mode='bilinear', align_corners=True)

#         x2 = self.aspp1(x)
#         x3 = self.aspp2(x)
#         x4 = self.aspp3(x)
#         x5 = self.aspp4(x)
        
#         print("Shapes: x1:", x1.shape, "x2:", x2.shape, "x3:", x3.shape, "x4:", x4.shape, "x5:", x5.shape)

#         x = torch.cat((x1, x2, x3, x4, x5), dim=1)
#         return self.concat_projection(x)

# # Assuming the input channels are 1024
# in_channels = 1024
# aspp = AtrousSpatialPyramidPooling(in_channels)

# # Save the model's state dictionary
# torch.save(aspp.state_dict(), 'aspp_model.pth')

# import tarfile

# # Create a tar file containing the saved model
# with tarfile.open('aspp_model.tar', 'w') as tar:
#     tar.add('aspp_model.pth', arcname='aspp_model.pth')

import torch
import torch.nn as nn
import torch.nn.functional as F




class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(AtrousSpatialPyramidPooling, self).__init__()
        out_channels = 256
        self.aspp1 = ASPPModule(in_channels, out_channels, dilation=1)
        self.aspp2 = ASPPModule(in_channels, out_channels, dilation=6)
        self.aspp3 = ASPPModule(in_channels, out_channels, dilation=12)
        self.aspp4 = ASPPModule(in_channels, out_channels, dilation=18)

        self.concat_projection = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.concat_projection(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Example usage
in_channels = 3  # CIFAR10 images have 3 channels
aspp = AtrousSpatialPyramidPooling(in_channels)
print(aspp)

# Example random input
input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 pixels
output = aspp(input_tensor)
print(output.shape)  # Should be [1, 10] for CIFAR10 classes





