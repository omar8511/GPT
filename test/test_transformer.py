import os
import pytest, torch

from tokeniser.tokeniser import Tokeniser
from tokeniser.preprocessor import PreProcessor
from tokeniser.bpe import BPE
from tokeniser.trainer import Trainer
from transformer.config import Config
from transformer.gpt import GPT
from transformer.train import train
from transformer.generate import generate
from transformer.kvcache import KVCache

@pytest.fixture(scope="module")
def gpt_and_tokeniser():
    sampleText = "hello world, this is a small sample corpus for testing the tokeniser!"
    config = Config(numMerges=25, dModel=32, numHeads=2, N=1, dFF=64, maxLen=16)
    preprocessor = PreProcessor()
    trainer = Trainer([preprocessor.tokenise(sampleText)], config)
    merges, tokens = trainer.train_BPE()
    tokens.update(preprocessor.byteEncoder.values())
    bpe = BPE(merges, tokens)
    config.vocabSize = len(tokens)

    tokeniser = Tokeniser(preprocessor, bpe)
    gpt = GPT(config)
    return gpt, tokeniser, config

def test_e2e_gpt(gpt_and_tokeniser):
    gpt, tokeniser, config = gpt_and_tokeniser
    encoded = tokeniser.encode("Hello")
    sequence = [tokenId for word in encoded for tokenId in word]
    output = gpt(torch.tensor([sequence]))
    assert output.shape == (1, len(sequence), config.vocabSize)

def test_causal_mask_no_leakage():
    config = Config(vocabSize=20, dModel=32, numHeads=2, N=1, dFF=64, maxLen=16)
    gpt = GPT(config)
    gpt.eval()  # dropout is stochastic per position and would break this comparison otherwise

    seqA = [3, 7, 12, 5, 9]
    seqB = [3, 7, 12, 1, 15]   
    divergeIdx = 3

    output = gpt(torch.tensor([seqA, seqB]))

    # assert that positions 1 -> Diverge index are all the same 
    # Self attention is allowed so allow i == j 

    assert torch.allclose(output[0, :divergeIdx, :], output[1, :divergeIdx, :])

def test_train_smoke(tmp_path):
    config = Config(numMerges=25, dModel=32, numHeads=2, N=1, dFF=64, maxLen=16, totalSteps=5, batchSize=8, evalEvery=5, warmupSteps=1)
    checkpointPath = str(tmp_path / "checkpoint.pt")

    gpt, tokeniser = train("tokeniser/test.txt", config, checkpointPath)

    assert os.path.isfile(checkpointPath)

def test_generate_e2e(gpt_and_tokeniser):
    gpt, tokeniser, config = gpt_and_tokeniser
    result = generate(gpt, tokeniser, "hi", 3, config)

    assert isinstance(result, str)
    assert len(result) > 0
def test_kv_cache_matches_full_recompute():
    # N > 1 so a bug that hands every block the same cache slot would show up here
    config = Config(vocabSize=60, dModel=64, numHeads=4, N=3, dFF=128, maxLen=32)
    gpt = GPT(config)
    gpt.eval()  # dropout is stochastic, so the two paths could never match with it on

    sequence = [5, 9, 14, 2, 31, 7, 22, 3, 41, 18]
    promptLen = 4

    with torch.no_grad():
        cache = [KVCache() for _ in range(config.N)]

        # prefill the prompt in one pass, then feed one token per step
        cachedLogits = [gpt(torch.tensor([sequence[:promptLen]]), cache=cache)[0, -1, :]]
        for i in range(promptLen, len(sequence)):
            cachedLogits.append(gpt(torch.tensor([[sequence[i]]]), cache=cache)[0, -1, :])

        for step, cached in enumerate(cachedLogits):
            # the uncached path recomputes the whole prefix, which is what the cache must reproduce
            full = gpt(torch.tensor([sequence[:promptLen + step]]))[0, -1, :]

            # loose enough for float32 reassociation from torch.cat, tight enough to catch a wrong offset
            # asserting per step matters: step 0 alone failing means prefill, later steps mean the offset
            assert torch.allclose(full, cached, atol=1e-5), f"cache diverges at decode step {step}"

def test_kv_cache_grows_by_one_per_decode_step():
    config = Config(vocabSize=60, dModel=64, numHeads=4, N=3, dFF=128, maxLen=32)
    gpt = GPT(config)
    gpt.eval()

    sequence = [5, 9, 14, 2]

    with torch.no_grad():
        cache = [KVCache() for _ in range(config.N)]

        gpt(torch.tensor([sequence]), cache=cache)
        # every layer holds one entry per prompt token
        assert all(slot.length() == len(sequence) for slot in cache)

        gpt(torch.tensor([[7]]), cache=cache)
        # exactly one - re-feeding the whole sequence each step would add len(sequence) instead
        assert all(slot.length() == len(sequence) + 1 for slot in cache)
