# Use OpenJDK 8 JRE (smaller than JDK since we're not compiling)
FROM openjdk:8-jre-alpine

# Install wget, unzip, curl, and Xvfb for virtual display
RUN apk add --no-cache wget unzip curl jq xvfb

# Set working directory
WORKDIR /app

# Download and extract the latest PDV release
RUN LATEST_URL=$(curl -s https://api.github.com/repos/wenbostar/PDV/releases/latest | jq -r '.assets[] | select(.name | endswith(".zip")) | .browser_download_url') && \
    echo "Downloading from: $LATEST_URL" && \
    wget -O pdv.zip "$LATEST_URL" && \
    unzip pdv.zip && \
    echo "Contents after unzip:" && \
    ls -la && \
    EXTRACTED_DIR=$(ls -d PDV-*/ | head -1) && \
    mv ${EXTRACTED_DIR}* . && \
    rm -rf "$EXTRACTED_DIR" pdv.zip && \
    echo "Final contents:" && \
    ls -la

# Create output directory
RUN mkdir -p /app/output

# Set environment for virtual display
ENV DISPLAY=:99

# Create a wrapper script that starts Xvfb and runs PDV
RUN echo '#!/bin/sh' > /app/pdv-cli.sh && \
    echo 'echo "Starting PDV CLI with virtual display..."' >> /app/pdv-cli.sh && \
    echo '# Start Xvfb in background' >> /app/pdv-cli.sh && \
    echo 'Xvfb :99 -screen 0 1024x768x16 &' >> /app/pdv-cli.sh && \
    echo 'XVFB_PID=$!' >> /app/pdv-cli.sh && \
    echo 'sleep 2  # Give Xvfb time to start' >> /app/pdv-cli.sh && \
    echo '' >> /app/pdv-cli.sh && \
    echo '# Run PDV with the provided arguments' >> /app/pdv-cli.sh && \
    echo 'java -jar PDV-*.jar "$@"' >> /app/pdv-cli.sh && \
    echo 'EXIT_CODE=$?' >> /app/pdv-cli.sh && \
    echo '' >> /app/pdv-cli.sh && \
    echo '# Clean up Xvfb' >> /app/pdv-cli.sh && \
    echo 'kill $XVFB_PID 2>/dev/null || true' >> /app/pdv-cli.sh && \
    echo 'exit $EXIT_CODE' >> /app/pdv-cli.sh && \
    chmod +x /app/pdv-cli.sh

# Use the wrapper script as entrypoint
ENTRYPOINT ["/app/pdv-cli.sh"]

# Default command shows help
CMD ["-h"]