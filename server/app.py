import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from transformer.checkpoint import loadCheckpoint
from transformer.generate import generate

# The full fine-tune conditions on the instruction more strongly than the rank-8 LoRA.
# serve.pt is sft.pt with the optimiser moments dropped - 113MB rather than 339MB
CHECKPOINT = os.environ.get("CHECKPOINT", "trained_model/serve.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the model once at startup rather than per request - the weights are ~113MB, so
    loading per call would allocate a fresh copy for every request in flight.
    """
    gpt, tokeniser, config = loadCheckpoint(CHECKPOINT)

    # loadCheckpoint returns a freshly constructed module, which defaults to train mode
    gpt.eval()

    app.state.gpt = gpt
    app.state.tokeniser = tokeniser
    app.state.config = config

    yield


app = FastAPI(
    title="mini-gpt",
    description="A from-scratch GPT trained on TinyStories, instruction tuned and served with a KV cache",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    # temperature must be > 0: zero divides the logits by zero and multinomial then throws
    # k is bounded against vocabSize at request time, since that comes from the checkpoint
    prompt: str = Field(min_length=1, max_length=4000)
    maxTokens: int = Field(default=100, ge=1, le=1024)
    temperature: float = Field(default=1.0, gt=0.0, le=2.0)
    k: int = Field(default=10, ge=1)


class GenerateResponse(BaseModel):
    text: str
    promptTokens: int
    truncated: bool
    contextLimit: int


@app.get("/health")
def health(request: Request) -> dict:
    """
    Reports what the process actually loaded, so a deploy can be checked without generating
    """
    gpt, config = request.app.state.gpt, request.app.state.config

    return {
        "status": "ok",
        "checkpoint": CHECKPOINT,
        "device": config.device,
        "parameters": sum(p.numel() for p in gpt.parameters()),
        "contextLimit": config.maxLen,
        "vocabSize": config.vocabSize,
    }


@app.post("/generate", response_model=GenerateResponse)
def generateStory(req: GenerateRequest, request: Request) -> GenerateResponse:
    """
    Generates a story from an instruction

    The SEP delimiter is appended here rather than inside generate, so generate stays a
    plain continuation function for the CLI while the API speaks the SFT format the model
    was fine tuned on.
    """
    gpt = request.app.state.gpt
    tokeniser = request.app.state.tokeniser
    config = request.app.state.config

    if req.k > config.vocabSize:
        # torch.topk cannot take more entries than the tensor holds
        raise HTTPException(status_code=400, detail=f"k must be <= vocabSize ({config.vocabSize})")

    if req.maxTokens > config.maxLen:
        raise HTTPException(status_code=400, detail=f"maxTokens must be <= contextLimit ({config.maxLen})")

    # count the caller's own text, not the delimited prompt - SEP always contributes tokens,
    # so checking the combined string would make this guard unreachable
    if sum(len(word) for word in tokeniser.encode(req.prompt)) == 0:
        # a non-empty string can still tokenise to nothing, and an empty tensor breaks the embedder
        raise HTTPException(status_code=400, detail="prompt contains no encodable tokens")

    prompt = req.prompt + tokeniser.preProcessor.SEP
    promptTokens = sum(len(word) for word in tokeniser.encode(prompt))

    text = generate(gpt, tokeniser, prompt, req.maxTokens, config, req.temperature, req.k)

    return GenerateResponse(
        text=text,
        promptTokens=promptTokens,
        # generate keeps the last maxLen tokens, so an over-long prompt is silently reshaped
        # unless the caller is told - the returned text will not start with what was sent
        truncated=promptTokens > config.maxLen,
        contextLimit=config.maxLen,
    )
