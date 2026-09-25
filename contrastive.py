
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ContrastiveEncoder(nn.Module):
    """
    Neural network encoder that converts the environment state
    into a compact representation.
    """

    def __init__(
        self,
        input_size,
        representation_size=64,
        hidden_size=128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, representation_size),
        )

    def forward(self, state):
        """
        Generate a normalized representation.
        """

        representation = self.encoder(state)

        representation = F.normalize(
            representation,
            p=2,
            dim=-1,
        )

        return representation


class ProjectionHead(nn.Module):
    """
    Projection head used during contrastive learning.
    """

    def __init__(
        self,
        representation_size=64,
        projection_size=32,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(
                representation_size,
                representation_size,
            ),
            nn.ReLU(),
            nn.Linear(
                representation_size,
                projection_size,
            ),
        )

    def forward(self, representation):
        projection = self.projection(
            representation
        )

        return F.normalize(
            projection,
            p=2,
            dim=-1,
        )


class ContrastiveLearner:
    """
    Contrastive representation learning model.

    Two related views of the same state are treated as a
    positive pair, while representations from other samples
    form negative pairs.
    """

    def __init__(
        self,
        input_size,
        representation_size=64,
        projection_size=32,
        hidden_size=128,
        learning_rate=1e-3,
        temperature=0.07,
    ):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.temperature = temperature

        self.encoder = ContrastiveEncoder(
            input_size=input_size,
            representation_size=representation_size,
            hidden_size=hidden_size,
        ).to(self.device)

        self.projection_head = ProjectionHead(
            representation_size=representation_size,
            projection_size=projection_size,
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.projection_head.parameters()),
            lr=learning_rate,
        )

    def encode(self, state):
        """
        Convert a state into a learned representation.
        """

        if isinstance(state, np.ndarray):
            state = torch.tensor(
                state,
                dtype=torch.float32,
            )

        if state.dim() == 1:
            state = state.unsqueeze(0)

        state = state.to(self.device)

        with torch.no_grad():
            representation = self.encoder(
                state
            )

        return representation

    def contrastive_loss(
        self,
        view_1,
        view_2,
    ):
        """
        Calculate InfoNCE-style contrastive loss.

        view_1 and view_2 contain two augmented views
        of the same batch of states.
        """

        z1 = self.projection_head(
            self.encoder(view_1)
        )

        z2 = self.projection_head(
            self.encoder(view_2)
        )

        similarity_matrix = torch.matmul(
            z1,
            z2.T,
        )

        similarity_matrix = (
            similarity_matrix
            / self.temperature
        )

        batch_size = z1.size(0)

        labels = torch.arange(
            batch_size,
            device=self.device,
        )

        loss_1 = F.cross_entropy(
            similarity_matrix,
            labels,
        )

        loss_2 = F.cross_entropy(
            similarity_matrix.T,
            labels,
        )

        loss = (
            loss_1 + loss_2
        ) / 2.0

        return loss

    def train_step(
        self,
        view_1,
        view_2,
    ):
        """
        Perform one contrastive learning step.
        """

        if isinstance(view_1, np.ndarray):
            view_1 = torch.tensor(
                view_1,
                dtype=torch.float32,
            )

        if isinstance(view_2, np.ndarray):
            view_2 = torch.tensor(
                view_2,
                dtype=torch.float32,
            )

        view_1 = view_1.to(self.device)
        view_2 = view_2.to(self.device)

        loss = self.contrastive_loss(
            view_1,
            view_2,
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(
                self.projection_head.parameters()
            ),
            max_norm=1.0,
        )

        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":

    print("Contrastive Representation Test")
    print("================================")

    input_size = 153
    representation_size = 64
    batch_size = 16

    learner = ContrastiveLearner(
        input_size=input_size,
        representation_size=representation_size,
    )

    print(f"Input state size: {input_size}")
    print(
        f"Representation size: "
        f"{representation_size}"
    )
    print(f"Batch size: {batch_size}")
    print(f"Device: {learner.device}")

    # Generate example state batch.
    states = np.random.rand(
        batch_size,
        input_size,
    ).astype(np.float32)

    # Create two slightly different views.
    view_1 = states + np.random.normal(
        0,
        0.01,
        states.shape,
    ).astype(np.float32)

    view_2 = states + np.random.normal(
        0,
        0.01,
        states.shape,
    ).astype(np.float32)

    loss = learner.train_step(
        view_1,
        view_2,
    )

    representation = learner.encode(
        states[0]
    )

    print(
        f"Contrastive loss: "
        f"{loss:.6f}"
    )

    print(
        f"Encoded representation shape: "
        f"{tuple(representation.shape)}"
    )

    print()
    print(
        "Contrastive representation test "
        "completed successfully."
    )

