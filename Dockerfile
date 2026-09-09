FROM python:3.13-slim

# Spaces runs containers as UID 1000 with a read-only filesystem outside the user's home
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Requirements first so a code change does not reinstall torch on every build
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user transformer/ transformer/
COPY --chown=user tokeniser/ tokeniser/
COPY --chown=user datastructures/ datastructures/
COPY --chown=user server/ server/
COPY --chown=user trained_model/serve.pt trained_model/serve.pt

ENV CHECKPOINT=trained_model/serve.pt

# 0.0.0.0 not 127.0.0.1, or the port is unreachable from outside the container.
# Cloud Run injects $PORT and requires the container to listen on it, so this is shell
# form rather than exec form - the exec array does no variable expansion.
EXPOSE 8080
CMD exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8080}
