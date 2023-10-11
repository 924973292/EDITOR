import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def display_image(image_path, mode=1):
    pre_fix = '/15127306268/wyh/MM/data/RGBNT201/train_171/'
    if mode == 1:
        pre_fix = pre_fix + 'RGB/'
    elif mode == 2:
        pre_fix = pre_fix + 'NI/'
    elif mode == 3:
        pre_fix = pre_fix + 'TI/'
    image = Image.open(pre_fix + image_path)
    resized_image = image.resize((128, 256))  # Resize to 256x128
    plt.imshow(resized_image)
    plt.axis('off')
    plt.show()


# Visualize the mask on the image
def visualize_multiple_masks(images, masks, mode, pre_fix, writer=None, epoch=None):
    num_images_to_display = 12  # Number of images to display
    images = images[:num_images_to_display]
    masks = masks[:num_images_to_display]
    num_rows = 2  # Number of rows in the display grid
    num_cols = 6  # Number of columns in the display grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    for i in range(num_images_to_display):
        # Reshape the mask to 16x8
        mask_2d = masks[i].reshape(16, 8).cpu().numpy()

        # Upscale the mask to 256x128
        mask_upscaled = np.kron(mask_2d, np.ones((16, 16)))

        # Append the appropriate mode prefix
        if mode == 1:
            prefix = pre_fix + 'RGB/'
        elif mode == 2:
            prefix = pre_fix + 'NI/'
        elif mode == 3:
            prefix = pre_fix + 'TI/'

        # Load the original image
        image = Image.open(prefix + images[i])
        original_image = image.resize((128, 256))  # Resize to 256x128

        # Convert the image to numpy array
        original_np = np.array(original_image)

        # Apply a color to the mask (e.g., yellow)
        mask_color = np.array([0, 0, 0])  # Black color for the mask
        masked_image = np.where(mask_upscaled[..., None], original_np, mask_color)

        row = i // num_cols
        col = i % num_cols

        # Display the masked image
        axes[row, col].imshow(masked_image)
        axes[row, col].axis('off')
    plt.tight_layout()
    if writer is not None:
        if mode == 1:
            sign = 'RGB'
        elif mode == 2:
            sign = 'NIR'
        elif mode == 3:
            sign = 'TIR'
        writer.add_figure('Person_Token_Select_' + sign, fig, global_step=epoch)
    # plt.close(fig)
    # plt.show()


class Person_Token_Select(nn.Module):
    def __init__(self, dim, ratio=0.5):
        super(Person_Token_Select, self).__init__()
        self.ratio = ratio

    def forward(self, features, img_path, mode=1, writer=None, epoch=None):
        # features: (batch_size, N+1, D)

        # Extract cls token and patch tokens
        cls_token = features[:, 0, :]  # cls token
        patch_tokens = features[:, 1:, :]  # patch tokens
        N = patch_tokens.shape[1]  # number of patch tokens
        # Calculate similarity between cls token and patch tokens
        similarity_scores = torch.mean(patch_tokens, dim=-1)
        # similarity_scores = (cls_token.unsqueeze(1) @ patch_tokens.transpose(1, 2)).squeeze()

        # Sort tokens by similarity score and select top k
        _, topk_indices = torch.topk(similarity_scores, int(N * self.ratio), dim=1)

        # Sort the topk_indices to maintain original order
        topk_indices = torch.sort(topk_indices, dim=1).values
        # Create a mask for selected tokens
        selected_tokens_mask = torch.zeros_like(similarity_scores, dtype=torch.bool)
        selected_tokens_mask.scatter_(1, topk_indices, 1)

        # Apply the mask to the patch tokens
        filtered_tokens = patch_tokens * selected_tokens_mask.unsqueeze(-1)
        if self.training:
            pre_fix = '/15127306268/wyh/MM/data/RGBNT201/train_171/'
            visualize_multiple_masks(img_path, selected_tokens_mask, mode=mode, pre_fix=pre_fix, writer=writer,
                                     epoch=epoch)
        else:
            pre_fix = '/15127306268/wyh/MM/data/RGBNT201/test/'
            # visualize_multiple_masks(img_path, selected_tokens_mask, mode=mode, pre_fix=pre_fix)
        if self.training:
            return filtered_tokens, selected_tokens_mask
        else:
            return filtered_tokens


class Person_Token_SelectC(nn.Module):
    def __init__(self, dim, ratio=0.5):
        super(Person_Token_Select, self).__init__()
        self.ratio = ratio

    def forward(self, features, img_path, mode=1):
        # features: (batch_size, N+1, D)

        # Extract cls token and patch tokens
        cls_token = features[:, 0, :]  # cls token
        patch_tokens = features[:, 1:, :]  # patch tokens
        N = patch_tokens.shape[1]  # number of patch tokens
        # Calculate similarity between cls token and patch tokens
        # similarity_scores = torch.mean(patch_tokens, dim=-1)
        similarity_scores = (cls_token.unsqueeze(1) @ patch_tokens.transpose(1, 2)).squeeze()

        # Sort tokens by similarity score and select top k
        _, topk_indices = torch.topk(similarity_scores, int(N * self.ratio), dim=1)

        # Sort the topk_indices to maintain original order
        topk_indices = torch.sort(topk_indices, dim=1).values
        # Create a mask for selected tokens
        selected_tokens_mask = torch.zeros_like(similarity_scores, dtype=torch.bool)
        selected_tokens_mask.scatter_(1, topk_indices, 1)

        # Apply the mask to the patch tokens
        filtered_tokens = patch_tokens * selected_tokens_mask.unsqueeze(-1)
        if self.training:
            pre_fix = '/15127306268/wyh/MM/data/RegDB/train/'
        else:
            pre_fix = '/15127306268/wyh/MM/data/RegDB/test/'
        visualize_multiple_masks(img_path, selected_tokens_mask, mode=mode, pre_fix=pre_fix)
        if self.training:
            return filtered_tokens, selected_tokens_mask
        else:
            return filtered_tokens
