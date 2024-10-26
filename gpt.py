import tensorflow as tf
import numpy as np

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

tf.random.set_seed(1337)

# Load data
with open('/content/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = tf.convert_to_tensor(encode(text), dtype=tf.int32)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = tf.random.uniform((batch_size,), maxval=len(data) - block_size, dtype=tf.int32)
    x = tf.stack([data[i:i + block_size] for i in ix])
    y = tf.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

class MultiHeadAttention(tf.keras.layers.Layer):
    """ Multi-head self-attention in parallel """

    def __init__(self, num_heads, head_size, output_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.output_dim = output_dim
        self.key_dense = tf.keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.query_dense = tf.keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.value_dense = tf.keras.layers.Dense(num_heads * head_size, use_bias=False)
        # Make sure output dimension matches input embedding size
        self.output_dense = tf.keras.layers.Dense(output_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.tril = tf.constant(np.tril(np.ones((block_size, block_size))), dtype=tf.float32)

    def call(self, x, training):
        batch_size, time_steps, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        keys = tf.reshape(self.key_dense(x), (batch_size, time_steps, self.num_heads, self.head_size))
        queries = tf.reshape(self.query_dense(x), (batch_size, time_steps, self.num_heads, self.head_size))
        values = tf.reshape(self.value_dense(x), (batch_size, time_steps, self.num_heads, self.head_size))

        keys = tf.transpose(keys, perm=[0, 2, 1, 3])
        queries = tf.transpose(queries, perm=[0, 2, 1, 3])
        values = tf.transpose(values, perm=[0, 2, 1, 3])

        attn_logits = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_size, tf.float32))
        
        mask = self.tril[:time_steps, :time_steps]
        mask = tf.reshape(mask, (1, 1, time_steps, time_steps))
        attn_logits = tf.where(mask == 0, -1e9, attn_logits)
        
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        
        attn_output = tf.matmul(attn_weights, values)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (batch_size, time_steps, self.num_heads * self.head_size))
        
        # Adjusted output to match embedding size
        return self.output_dense(attn_output)

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, n_embd):
        super().__init__()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * n_embd, activation='relu'),
            tf.keras.layers.Dense(n_embd),
            tf.keras.layers.Dropout(dropout)
        ])

    def call(self, x, training=False):
        return self.ffn(x, training=training)

class TransformerBlock(tf.keras.layers.Layer):
    """ Transformer block with attention and feed-forward network """

    def __init__(self, n_embd, n_head):
        super().__init__()
        # Pass n_embd as output_dim to match the model's embedding size
        self.attention = MultiHeadAttention(num_heads=n_head, head_size=n_embd // n_head, output_dim=n_embd)
        self.ffn = FeedForward(n_embd)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        x = x + self.attention(self.ln1(x), training=training)
        x = x + self.ffn(self.ln2(x), training=training)
        return x


class GPTLanguageModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, n_embd)
        self.position_embedding = tf.keras.layers.Embedding(block_size, n_embd)
        self.transformer_blocks = [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.head = tf.keras.layers.Dense(vocab_size)

    def call(self, x, training=False):
        seq_length = tf.shape(x)[1]
        pos = tf.range(0, seq_length, dtype=tf.int32)[tf.newaxis, :]
        x = self.token_embedding(x) + self.position_embedding(pos)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, start_tokens, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(start_tokens)
            logits = logits[:, -1, :]
            next_token = tf.random.categorical(logits, num_samples=1)
            start_tokens = tf.concat([start_tokens, next_token], axis=-1)
        return start_tokens

# Instantiate the model
model = GPTLanguageModel()
optimizer = tf.keras.optimizers.Adam(learning_rate)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for iteration in range(max_iters):
    if iteration % eval_interval == 0:
        # Evaluation step can be implemented similarly to the training step
        print(f"Step {iteration}: Training model...")

    x_batch, y_batch = get_batch('train')
    loss = train_step(x_batch, y_batch)
    print(f"Iteration {iteration}: Loss = {loss:.4f}")

# Generate text
start_tokens = tf.constant([[stoi[' ']]], dtype=tf.int32)
generated = model.generate(start_tokens, max_new_tokens=500)
print(decode(generated[0].numpy().tolist()))
