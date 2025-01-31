from typing import Optional, Union
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

class ReluSquared(torch.nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

def get_activation(activation):
    """
    In order to pickle our module, it is preferable not to pass modules as parameters (even though it should also work).

    Args:
        activation (str): Activation function

    Returns:
        torch.nn.Module: Activation function
    """
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "relu_squared":
        return ReluSquared()
    elif activation == "gelu":
        return torch.nn.GELU()
    else:
        return torch.nn.Identity()

class FFN(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        mult: int = 4,
        dim_hidden: Optional[int] = None,
        activation: str = "relu",
        bias: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        """Simple FFN/MLP in style of the inverted-bottleneck for Transformers

        If dim_hidden is None, it is set to dim_in * mult.
        If dim_hidden is 0, there is only 1 layer!

        Args:
            dim_in (int): The embedding dimension of the input (last dimension)
            dim_hidden (int): The interior embedding dimension of the inverted bottleneck
            dim_out (int): The embedding dimension of the output (last dimension)
            activation (Union[torch.nn.Module, Callable], optional): Activation to use. Defaults to F.relu.
            bias (bool, optional): Whether to use a bias in the first layer. Defaults to True.
            dropout_p (float, optional): Dropout. Defaults to 0.0.
        """
        super().__init__()
        self.dim_in = dim_in
        if dim_hidden is None and mult > 0:
            self.dim_hidden = int(dim_in * mult)
        elif dim_hidden is None and mult == 0:
            self.dim_hidden = 0
        else:
            self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.activation = get_activation(activation)
        if self.dim_hidden == 0:
            self.ff1 = torch.nn.Linear(dim_in, self.dim_out, bias=bias)
            self.ff2 = torch.nn.Identity()
        else:
            self.ff1 = torch.nn.Linear(dim_in, self.dim_hidden, bias=bias)
            self.ff2 = torch.nn.Linear(self.dim_hidden, dim_out)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.ff2(self.activation(self.ff1(x))))

class EmbedFeaturesFT(torch.nn.Module):
    def __init__(
        self,
        nr_categories: list,
        dim_embedding: int,
        nr_cont_features: int,
        nr_cat_features: int,
        internal_dimension: int = 0,
        cont_embd_act: str = "relu",
        cont_emb_dropout_p: float = 0.0,
        num_special_tokens: int = 0,
        nr_vec_features: int = 0,
        dim_vec_features: int = 0,
    ) -> None:
        """Embed continuous and categorical features for tabular data.

        Args:
            nr_categories (list): A list of the number of categories (possible elements) for each categorical column
            dim_embedding (int): The embedding dimensions of both feature types
            nr_cont_features, nr_cat_features (int): The number of each feature type
            internal_dimension (int, optional): Hidden dimension of the MLPs doing the
                embedding for continuous features. Defaults to 100.
            cont_embd_act (Union[torch.nn.Module, Callable], optional):
                Activation of the MLPs doing the embedding for continuous features. Defaults to F.relu.
            cont_emb_dropout_p (float, optional):
                Dropout for the MLPs doing the embedding for continuous features. Defaults to 0.0.
            num_special_tokens (int, optional): Number of additional special tokens to include. Defaults to 0.
        """
        super().__init__()
        # Add Special Tokens if needed
        self.dim_embedding = dim_embedding
        
        if not (nr_cat_features == 0):
            categories_offset = F.pad(torch.tensor(list(nr_categories)), (1, 0), value=num_special_tokens)
            
            # Get starting offsets for embeddings
            # Essentially, we will offset every numerical category to ensure that every feature has unique "values"
            categories_offset = categories_offset.cumsum(dim=-1)[:-1] 

            total_cat_embeddings = sum(nr_categories) + categories_offset.max()

            # Categorical Embeddings
            self.cat_embed = torch.nn.Embedding(total_cat_embeddings, dim_embedding)
        else:
            self.cat_embed = None
        
        if not (nr_cont_features == 0):
            # MLP Embedding for cont
            self.cont_embed = torch.nn.ModuleList(
                [
                    FFN(
                        dim_in=1,
                        dim_hidden=internal_dimension,
                        dim_out=dim_embedding,
                        mult=0,
                        activation=cont_embd_act,
                        dropout_p=cont_emb_dropout_p,
                    )
                    for _ in range(0, nr_cont_features)
                ]
            )
        else:
            self.cont_embed = None
        
        if not (nr_vec_features == 0):
            # MLP Embedding for vec
            self.vec_embed = FFN(
                dim_in=dim_vec_features,
                dim_hidden=internal_dimension,
                dim_out=dim_embedding,
                mult=0,
                activation=cont_embd_act,
                dropout_p=cont_emb_dropout_p,
            )
        else:
            self.vec_embed = None


    def _embed_vec(self, x_vec: torch.Tensor) -> torch.Tensor:
        if x_vec is None:
            return None
        return self.vec_embed(x_vec)

    def _embed_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat is None:
            return None
        return self.cat_embed(x_cat + self.categories_offset)

    def _embed_cont(self, x_cont: torch.Tensor) -> torch.Tensor:
        if x_cont is None:
            return None
        x_cont_embed = torch.empty(
            size=(x_cont.shape[0], x_cont.shape[1], int(self.dim_embedding)),
            device=x_cont.device,
        )
        for idx, mlp in enumerate(self.cont_embed):
            x_cont_embed[..., idx, :] = mlp(x_cont[..., idx].unsqueeze(1))
        return x_cont_embed

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor, x_vec: torch.Tensor) -> tuple:
        return self._embed_categorical(x_cat), self._embed_cont(x_cont), self._embed_vec(x_vec)
    
    
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        dim_input: int,
        nr_heads: int = 8,
        dropout_p: float = 0.0,
        scale_factor: float = 0.5,
    ):
        """Ye olde Multihead Attention, implmented with Einstein Notation.
        Note: There' ain't no masking here, so be careful!

        Args:
            dim_input (int): The input dimension
            nr_heads (int, optional): Number of heads. Defaults to 8.
            dropout_p (float, optional): Dropout. Defaults to 0.0.
            scale_factor (float, optional): Exponent of the scaling division - default is square root. Defaults to 0.5.
        """
        super().__init__()
        self.nr_heads = nr_heads
        self.dim_input = dim_input
        self.dim_head = dim_input // nr_heads
        self.scale = self.dim_head**-scale_factor

        self.to_qkv = torch.nn.Linear(dim_input, self.dim_head * self.nr_heads * 3, bias=False)

        self.to_out = torch.nn.Linear(self.dim_head * nr_heads, dim_input)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.nr_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # Add an extra dimension for the heads (b, 1, i, j)
            sim = sim.masked_fill(attn_mask == 0, float("-inf"))

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)
