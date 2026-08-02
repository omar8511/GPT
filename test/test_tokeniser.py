import pytest

from tokeniser.tokeniser import Tokeniser
from tokeniser.preprocessor import PreProcessor
from tokeniser.bpe import BPE
from tokeniser.trainer import Trainer
from transformer.config import Config

@pytest.fixture(scope="module")
def tokeniser():
    sampleText = "hello world, this is a small sample corpus for testing the tokeniser!"
    config = Config(numMerges=25)
    preprocessor = PreProcessor()
    trainer = Trainer([preprocessor.tokenise(sampleText)], config)
    merges, tokens = trainer.train_BPE()
    tokens.update(preprocessor.byteEncoder.values())
    bpe = BPE(merges, tokens)

    tokeniser = Tokeniser(preprocessor, bpe)
    return tokeniser

def test_e2e_simple(tokeniser):
    text = "hello world"
    assert tokeniser.decode(tokeniser.encode(text)) == text

def test_e2e_emojis(tokeniser):
    text = "😀🦄🌪️🍕🚀🐸🎸🌈🧩🔥🦊🍩⚡🌙🎲🐼💎🌻👾🍀🎯✨"
    assert tokeniser.decode(tokeniser.encode(text)) == text

def test_e2e_long(tokeniser):
    text = "Sparkle 🌟 midnight 🌙 river 🌊 cosmic 🚀 banana 🍌 thunder ⚡ whisper 🦊 galaxy 🌌 dream 🍀 rocketship 🛸 forest 🌲 magic 🪄 ocean 🐬 rainbow 🌈 shadow 🖤 crystal 💎 adventure 🗺️ fire 🔥 butterfly 🦋 coffee ☕ mystery 🧩 mountain ⛰️ lightning ⚡ cherry 🍒 planet 🪐 laughter 😂 golden ✨ dragon 🐉 cloud ☁️ melody 🎵 secret 🔮 flower 🌸 journey 🚲 moonlight 🌕 treasure 💰 fantasy 🧚‍♂️ oceanwave 🌊 star ⭐ cookie 🍪 jungle 🌴 energy ⚡ wonder 🌻 phoenix 🔥 harmony 🎶 puzzle 🧠 dreamscape 🌠"
    assert tokeniser.decode(tokeniser.encode(text)) == text

def test_e2e_complex(tokeniser):
    text = "A7!k#9_zQ@42m$Lp&*vX1+==helloWORLD123~[]{}<>✓★☂☃☀☯☮☾✦✧⚡🔥🌙🚀🦄💎🍀🎲🐉αβγδεζηθλμξπσφψωЖДФЯБ你好世界こんにちは안녕"
    assert tokeniser.decode(tokeniser.encode(text)) == text
    