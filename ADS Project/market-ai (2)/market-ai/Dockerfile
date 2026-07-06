# ============================================
# RAPIDS GPU BASE IMAGE
# ============================================
FROM rapidsai/base:24.02-cuda12.0-py3.10

# ============================================
# SET WORKDIR
# ============================================
WORKDIR /app

# ============================================
# SWITCH TO ROOT (for installations)
# ============================================
USER root

# ============================================
# SYSTEM DEPENDENCIES
# ============================================
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# COPY REQUIREMENTS FIRST (for caching)
# ============================================
COPY requirements.txt .

# ============================================
# INSTALL PYTHON DEPENDENCIES
# ============================================
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# COPY PROJECT FILES
# ============================================
COPY . .

# ============================================
# SWITCH BACK TO NON-ROOT (security best practice)
# ============================================
USER rapids

# ============================================
# EXPOSE DASHBOARD PORT
# ============================================
EXPOSE 8050

# ============================================
# DEFAULT COMMAND
# ============================================
CMD ["python", "main.py"]