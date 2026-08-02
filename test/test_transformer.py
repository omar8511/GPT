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
    output = gpt([sequence])
    assert output.shape == (1, len(sequence), config.vocabSize)

def test_causal_mask_no_leakage():
    config = Config(vocabSize=20, dModel=32, numHeads=2, N=1, dFF=64, maxLen=16)
    gpt = GPT(config)

    seqA = [3, 7, 12, 5, 9]
    seqB = [3, 7, 12, 1, 15]   
    divergeIdx = 3

    output = gpt([seqA, seqB])   

    # assert that positions 1 -> Diverge index are all the same 
    # Self attention is allowed so allow i == j 

    assert torch.allclose(output[0, :divergeIdx, :], output[1, :divergeIdx, :])

def test_train_smoke(tmp_path):
    config = Config(numMerges=25, dModel=32, numHeads=2, N=1, dFF=64, maxLen=16, numEpochs=1, batchSize=8)
    checkpointPath = str(tmp_path / "checkpoint.pt")

    gpt, tokeniser = train("tokeniser/test.txt", config, checkpointPath)

    assert os.path.isfile(checkpointPath)

def test_generate_e2e(gpt_and_tokeniser):
    gpt, tokeniser, config = gpt_and_tokeniser
    result = generate(gpt, tokeniser, "hi", 3, config)

    assert isinstance(result, str)
    assert len(result) > 0