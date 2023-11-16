class EncoderConfig:
    def __init__(self, input_dim, hidden_dim, z_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim


class DecoderConfig:
    def __init__(
        self,
        z_dim,
        conductor_hidden_dim,
        conductor_output_dim,
        decoder_hidden_dim,
        num_subsequences,
        subsequence_length,
        vocab_size,
    ):
        self.z_dim = z_dim
        self.conductor_hidden_dim = conductor_hidden_dim
        self.conductor_output_dim = conductor_output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_subsequences = num_subsequences
        self.subsequence_length = subsequence_length
        self.vocab_size = vocab_size
