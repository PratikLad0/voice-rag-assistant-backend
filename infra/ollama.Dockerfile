FROM ollama/ollama:latest
ENV OLLAMA_HOST=0.0.0.0
COPY start.sh /start.sh
RUN chmod +x /start.sh
EXPOSE 11434
CMD ["/start.sh"]
