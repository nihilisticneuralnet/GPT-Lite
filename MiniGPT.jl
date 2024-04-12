import Pkg
Pkg.add("CUDA")

using CUDA
if CUDA.functional()
    println("CUDA is functional.")
    device_count = CUDA.device_count()
    println("Number of CUDA devices: $device_count")
else
    println("CUDA is not functional. Please check your installation.")
end


using Flux
using Flux: @epochs, onehotbatch, mse, throttle
using CUDA

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = cpu() 
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

import Random
Random.seed!(1337)

text = ""
open("input.txt", "r") do f
    global text = read(f, String)
end

chars = sort(collect(Set(text)))
vocab_size = length(chars)

stoi = Dict(ch => i for (i, ch) in enumerate(chars))
itos = Dict(i => ch for (i, ch) in enumerate(chars))

encode(s) = [stoi[c] for c in s]
decode(l) = join([itos[i] for i in l])

data = Int64.(encode(text))
n = floor(Int, 0.9 * length(data))  
train_data = data[1:n]
val_data = data[n+1:end]

function get_batch(split)
    data = if split == "train"
        train_data
    else
        val_data
    end
    ix = rand(1:length(data) - block_size, batch_size)
    x = [data[i:i+block_size-1] for i in ix]
    y = [data[i+1:i+block_size] for i in ix]
    x, y = CUDA.CuArray(x), CUDA.CuArray(y)
    return x, y
end

model = Chain(
    LSTM(vocab_size, n_embd),
    LSTM(n_embd, n_embd),
    Dense(n_embd, vocab_size),
    logsoftmax
)

function estimate_loss()
    out = Dict{String, Float64}()
    Flux.eval!(model)
    for split in ["train", "val"]
        losses = zeros(eval_iters)
        for k in 1:eval_iters
            X, Y = get_batch(split)
            logits = model(X)
            loss = mse(logits, Y)
            losses[k] = Flux.item(loss)
        end
        out[split] = mean(losses)
    end
    Flux.train!(model)
    return out
end

struct Head
    key::Dense
    query::Dense
    value::Dense
    tril::CuArray{Float32}

    function Head(head_size)
        key = Dense(n_embd, head_size; bias=false)
        query = Dense(n_embd, head_size; bias=false)
        value = Dense(n_embd, head_size; bias=false)
        tril = CUDA.ones(Float32, block_size, block_size)
        tril = Flux.tril(tril)
        dropout = Dropout(dropout)

        new(key, query, value, tril)
    end
end

function (head::Head)(x)
    B, T, C = size(x)
    k = head.key(x)   # (B, T, hs)
    q = head.query(x) # (B, T, hs)
    wei = q * permutedims(k, (1, 3, 2)) * (size(k, 2)^-0.5) # (B, T, hs) * (B, hs, T) -> (B, T, T)
    wei[head.tril[T, T] .== 0] .= -Inf # (B, T, T)
    wei = softmax(wei, dims=3) # (B, T, T)
    wei = head.dropout(wei)
    v = head.value(x) # (B, T, hs)
    out = wei * v # (B, T, T) * (B, T, hs) -> (B, T, hs)
    return out
end

struct MultiHeadAttention
    heads::Vector{Head}
    proj::Dense
    dropout::Dropout

    function MultiHeadAttention(num_heads, head_size)
        heads = [Head(head_size) for _ in 1:num_heads]
        proj = Dense(head_size * num_heads, n_embd)
        dropout = Dropout(dropout)

        new(heads, proj, dropout)
    end
end

function (attention::MultiHeadAttention)(x)
    out = cat([h(x) for h in attention.heads]...; dims=3)
    out = attention.dropout(attention.proj(out))
    return out
end

struct FeedFoward
    net::Chain

    function FeedFoward(n_embd)
        net = Chain(
            Dense(n_embd, 4n_embd),
            relu,
            Dense(4n_embd, n_embd),
            Dropout(dropout)
        )

        new(net)
    end
end

function (ffn::FeedFoward)(x)
    return ffn.net(x)
end  

struct Block
    sa::MultiHeadAttention
    ffwd::FeedFoward
    ln1::LayerNorm
    ln2::LayerNorm

    function Block(n_embd, n_head)
        head_size = div(n_embd, n_head)
        sa = MultiHeadAttention(n_head, head_size)
        ffwd = FeedFoward(n_embd)
        ln1 = LayerNorm(n_embd)
        ln2 = LayerNorm(n_embd)

        new(sa, ffwd, ln1, ln2)
    end
end

function (block::Block)(x)
    x = x + block.sa(block.ln1(x))
    x = x + block.ffwd(block.ln2(x))
    return x
end

     

struct GPTLanguageModel
    token_embedding_table::Embedding
    position_embedding_table::Embedding
    blocks::Chain
    ln_f::LayerNorm
    lm_head::Dense
end

function _init_weights(model)
    for (name, param) in Flux.params(model)
        if isa(param, AbstractConv)
            param.weight.data .= Flux.glorot_normal(param.weight.data)
            if param.bias !== nothing
                param.bias.data .= 0
            end
        elseif isa(param, Embed)
            param.weight.data .= Flux.glorot_normal(param.weight.data)
        end
    end
end

function GPTLanguageModel()
    token_embedding_table = Embedding(vocab_size, n_embd)
    position_embedding_table = Embedding(block_size, n_embd)
    blocks = Chain([Block(n_embd, n_head) for _ in 1:n_layer]...)
    ln_f = LayerNorm(n_embd)
    lm_head = Dense(n_embd, vocab_size)

    model = GPTLanguageModel(token_embedding_table, position_embedding_table, blocks, ln_f, lm_head)
    _init_weights(model)

    return model
end

function (model::GPTLanguageModel)(idx; targets = nothing)
    B, T = size(idx)

    # idx and targets are both (B,T) tensor of integers
    tok_emb = model.token_embedding_table(idx) # (B,T,C)
    pos_emb = model.position_embedding_table(collect(1:T)) # (T,C)
    x = tok_emb + pos_emb # (B,T,C)
    x = model.blocks(x) # (B,T,C)
    x = model.ln_f(x) # (B,T,C)
    logits = model.lm_head(x) # (B,T,vocab_size)

    if isnothing(targets)
        loss = nothing
    else
        logits = Flux.reshape(logits, (B*T, vocab_size))
        targets = Flux.reshape(targets, (B*T,))
        loss = Flux.crossentropy(logits, targets)
    end

    return logits, loss
end

function generate(model::GPTLanguageModel, idx, max_new_tokens)
    # idx is (B, T) array of indices in the current context
    for _ in 1:max_new_tokens
        idx_cond = idx[:, end-block_size+1:end]
        logits, loss = model(idx_cond)
        logits = logits[:, end, :] # becomes (B, C)
        probs = Flux.softmax(logits, dims=2) # (B, C)
        idx_next = Flux.multinomial(probs, 1) # (B, 1)
        idx = hcat(idx, idx_next) # (B, T+1)
    end
    return idx
end

model = GPTLanguageModel()
m = model |> device
params_count = sum(p -> length(Flux.params(p)), Flux.children(m)) / 1e6
println("$(params_count) M parameters")

optimizer = ADAMW(Flux.params(m), learning_rate)

for iter in 1:max_iters
    if iter % eval_interval == 0 || iter == max_iters
        losses = estimate_loss()
        println("step (losses["train"]:.4f), val loss $(losses["val"]:.4f)")
    end

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    Flux.reset!(optimizer)
    Flux.back!(loss)
    Flux.update!(optimizer, Flux.params(m))
end

context = CUDA.zeros((1, 1), dtype=Int64) |> device
generated = generate(model, context, max_new_tokens=500)
println(decode(generated[1, :]))
