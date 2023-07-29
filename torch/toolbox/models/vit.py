import torch

from torch import nn


# 1. Subclass nn.Module
class PatchEmbedding(nn.Module):
    """It converts an 2D image input into a sequence of learnable embeddings.
    Example Usage:
        l = PatchEmbedding(in_channels=3, embedding_dim=768, patch_size=16)
    """

    def __init__(self, in_channels: int, embedding_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

        # 2. Initialize a convolution layer to create patch embeddings
        self.embedding_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # 3. Initialize a flatten layer to flatten 2D patches to 1D.
        self.flatten_layer = nn.Flatten(start_dim=-2, end_dim=-1)

    # 4. Define the forward pass
    def forward(self, x):
        # Validate input
        assert (
            x.shape[-1] % self.patch_size == 0
        ), f"Input image shape: {x.shape[-2:]} must be divisible by patch_size: {patch_size}"
        x = self.embedding_layer(x)
        x = self.flatten_layer(x)
        x = x.permute(0, 2, 1)

        return x


# 1. Create a class for MSA block
class MultiHeadSelfAttentionBlock(nn.Module):
    # 2. Initialize the module with hyperparameters
    def __init__(
        self, embedding_dim: int = 768, num_heads: int = 12, dropout: float = 0.0
    ):
        super().__init__()

        # 3. Initialize the normalization layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the MSA layer
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    # 5. Create forward pass
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attention(
            query=x, key=x, value=x, need_weights=False
        )

        return attn_output


# 1. Create a MLPBlock class that implements nn.Module
class MLPBlock(nn.Module):
    """Creates a Multilayer Perceptron Block for ViT

    It is structured as:
    [LayerNorm -> Linear -> Non-Linearity -> Dropout -> Linear -> Dropout]
    """

    # 2. Initialize the object with input hyperparameters
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()

        # 3. Create a Layer Norm
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create a Multilayer Perceptron (MLP) block
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_dim, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        )

    # 5. Implement the forward pass
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp_block(x)

        return x


# 1. Create a TransformerEncoderBlock class that implements nn.Module
class TransformerEncoderBlock(nn.Module):
    """Creates a transformer encoder block which assembles MSA and MLP blocks together.

    [MSABlock -> [MSABlock_output + MSABlock_input] -> MLPBlock -> [MLPBlock_output + MLPBlock_input]]
    """

    # 2. Initialize the class with the hyperparameters
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        num_heads: int = 12,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()

        # 3. Create a MSA block
        self.msa_block = MultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout
        )

        # 4. Create MLP block
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim, mlp_dim=mlp_dim, dropout=mlp_dropout
        )

    # 5. Implement the forward pass
    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x


# 1. Create a ViT class which inherits nn.Module.
class ViT(nn.Module):
    """An implementation of Vision Transformer"""

    # 2. Initialize model hyperparameters.
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embedding_dim: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        embedding_dropout: float = 0.1,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
        num_encoders: int = 12,
    ):
        super().__init__()

        # 3. Add assert to ensure that image is divisible into patches.
        assert (
            img_size % patch_size == 0
        ), f"img_size: {img_size} % patch_size: {patch_size} != 0"

        # 4. Compute the number of patches.
        self.num_patches = (img_size // patch_size) ** 2

        # 5. Create a learnable class embedding token.
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim))

        # 6. Create a learnable position embedding token.
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, embedding_dim)
        )

        # 7. Create a embedding dropout layer.
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 8. Create a patch embedding layer using PatchEmbedding module.
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels, embedding_dim=embedding_dim, patch_size=patch_size
        )

        # 9. Initialize a Sequential with a series of TransformerEncoderBlocks.
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(num_encoders)
            ]
        )

        # 10. Create a classifier head with a LayerNorm and a Linear layer.
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    # 11. Create a forward method.
    def forward(self, x):
        # 12. Get the batch size of the input.
        batch_size = x.shape[0]

        # 13. Create class token embedding for each element in the batch.
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # 14. Create patch embedding.
        x = self.patch_embedding(x)

        # 15. Attach class token embedding to the patch embedding.
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to the patch and class token embedding.
        x = self.position_embedding + x

        # 17. Pass the patch and position embedding through the dropout layer (Step 7).
        x = self.embedding_dropout(x)

        # 18. Pass the patch and position embeddings from step 16 through the stack of transformer encoders.
        x = self.encoder(x)

        # 19. Pass index 0 of the output of the stack of transformer encoders through the classifier head.
        x = self.classifier(x[:, 0])

        # 20. That's ViT for you!
        return x


# import random

# from toolbox import data_download, data_setup, engine, utils, evaluation, visualization
# from .. import utils

# from torchvision import transforms
# from matplotlib import pyplot as plt

if __name__ == "__main__":
    # 1. Import modules
    import torchinfo

    from torchinfo import summary as model_summary

    # 2. Print Version Info
    print(f"torch version: {torch.__version__}")
    print(f"torchinfo version: {torchinfo.__version__}")

    # 3. Initialize Hyperparameters
    SEED = 3
    NUM_ENCODERS = 1
    NUM_CLASSES = 10
    IMG_SIZE = 64
    PATCH_SIZE = 16

    # 4. Instantiate ViT
    vit = ViT(img_size=IMG_SIZE, num_encoders=NUM_ENCODERS, num_classes=NUM_CLASSES)

    # 5. Summarize ViT
    print(
        model_summary(
            model=vit,
            input_size=(1, 3, IMG_SIZE, IMG_SIZE),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
        )
    )

    print(f"\n{vit}")
