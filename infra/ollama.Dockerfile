FROM ollama/ollama:latest

# Make sure the API listens on the container's network interface
ENV OLLAMA_HOST=0.0.0.0

# Copy an entrypoint that pulls the model, then foregrounds the server
COPY start.sh /start.sh
RUN chmod +x /start.sh

# The persistent disk will be mounted at /root/.ollama by Render per render.yaml
EXPOSE 11434

CMD ["/start.sh"]
