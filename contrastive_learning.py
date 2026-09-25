import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """
    Neural network encoder for edge-cloud system states.

    Input:
        state vector

    Output:
        latent representation
    """

    def __init__(
        self,
        input_dim=153,
        hidden_dim=128,
        representation_dim=64
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, representation_dim)
        )

    def forward(self, state):
        """
        Convert system states into latent representations.
        """

        representation = self.network(state)

        # L2 normalisation is useful for cosine similarity.
        representation = F.normalize(
            representation,
            p=2,
            dim=1
        )

        return representation


class ProjectionHead(nn.Module):
    """
    Projection head used for contrastive learning.
    """

    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, representation):

        projection = self.network(
            representation
        )

        projection = F.normalize(
            projection,
            p=2,
            dim=1
        )

        return projection


class ContrastiveEncoder(nn.Module):
    """
    Complete contrastive representation-learning model.
    """

    def __init__(
        self,
        input_dim=153,
        hidden_dim=128,
        representation_dim=64,
        projection_dim=64
    ):
        super().__init__()

        self.encoder = StateEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            representation_dim=representation_dim
        )

        self.projector = ProjectionHead(
            input_dim=representation_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

    def forward(self, state):

        representation = self.encoder(
            state
        )

        projection = self.projector(
            representation
        )

        return representation, projection


def info_nce_loss(
    projection_1,
    projection_2,
    temperature=0.1
):
    """
    Calculate the InfoNCE contrastive loss.

    projection_1 and projection_2 contain two
    augmented views of the same batch of states.
    """

    if projection_1.shape != projection_2.shape:
        raise ValueError(
            "Both projections must have the same shape."
        )

    batch_size = projection_1.size(0)

    # Normalise projections
    projection_1 = F.normalize(
        projection_1,
        dim=1
    )

    projection_2 = F.normalize(
        projection_2,
        dim=1
    )

    # Combine both views
    representations = torch.cat(
        [
            projection_1,
            projection_2
        ],
        dim=0
    )

    # Similarity matrix
    similarity_matrix = torch.matmul(
        representations,
        representations.T
    )

    similarity_matrix = (
        similarity_matrix / temperature
    )

    # Remove self-similarity
    mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=similarity_matrix.device
    )

    similarity_matrix = similarity_matrix.masked_fill(
        mask,
        -float("inf")
    )

    # Positive-pair indices
    labels = torch.arange(
        batch_size,
        device=similarity_matrix.device
    )

    labels = torch.cat(
        [
            labels + batch_size,
            labels
        ]
    )

    loss = F.cross_entropy(
        similarity_matrix,
        labels
    )

    return loss


def augment_state(
    state,
    noise_std=0.01
):
    """
    Create a slightly perturbed view of a state.

    This is used to construct a positive pair.
    """

    noise = torch.randn_like(state) * noise_std

    augmented_state = state + noise

    return augmented_state


if __name__ == "__main__":

    print("Contrastive Learning Test")
    print("================================")

    # Configuration
    input_dim = 153
    batch_size = 16

    # Create model
    model = ContrastiveEncoder(
        input_dim=input_dim,
        hidden_dim=128,
        representation_dim=64,
        projection_dim=64
    )

    # Generate example states
    states = torch.randn(
        batch_size,
        input_dim
    )

    # Create a second view
    augmented_states = augment_state(
        states
    )

    # Forward pass
    representation_1, projection_1 = model(
        states
    )

    representation_2, projection_2 = model(
        augmented_states
    )

    # Contrastive loss
    loss = info_nce_loss(
        projection_1,
        projection_2
    )

    print(
        f"Input state shape       : "
        f"{states.shape}"
    )

    print(
        f"Representation shape    : "
        f"{representation_1.shape}"
    )

    print(
        f"Projection shape        : "
        f"{projection_1.shape}"
    )

    print(
        f"Contrastive loss        : "
        f"{loss.item():.6f}"
    )

    print("\nContrastive encoder test completed.")