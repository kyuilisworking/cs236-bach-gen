class EncoderConfig:
    def __init__(self, input_dim, hidden_dim, z_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim


class DecoderConfig:
    def __init__(
        self,
        decoder_type,
        *,
        z_dim=512,
        conductor_hidden_dim=1024,
        conductor_layers=1,
        embedding_dim=512,
        decoder_hidden_dim=1024,
        decoder_layers=2,
        num_subsequences,
        subsequence_length,
        vocab_size=129,
        teacher_force=True,
        teacher_force_prob=1.0,
    ):
        self.decoder_type = decoder_type
        self.z_dim = z_dim
        self.conductor_hidden_dim = conductor_hidden_dim
        self.conductor_layers = conductor_layers
        self.embedding_dim = embedding_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_layers = decoder_layers
        self.num_subsequences = num_subsequences
        self.subsequence_length = subsequence_length
        self.vocab_size = vocab_size
        self.teacher_force = teacher_force
        self.teacher_force_prob = teacher_force_prob


class VQVAEConfig:
    def __init__(self) -> None:
        pass
