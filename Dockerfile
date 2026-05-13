# Start with the official Ruby 3.3 image (Stable and highly compatible with Jekyll 4.x)
FROM ruby:3.3-bookworm

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set Bundler environment variables so the build AND run phases match
ENV BUNDLE_PATH=/usr/local/bundle
ENV BUNDLE_BIN=/usr/local/bundle/bin
ENV PATH=$BUNDLE_BIN:$PATH

# 1. Install curl to fetch the NodeSource setup script
# 2. Add the NodeSource repo for Node.js 22.x LTS
# 3. Install Node.js, Git, and build essentials
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y \
    nodejs \
    build-essential \
    zlib1g-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /site

# Update rubygems and install the latest bundler
RUN gem update --system && gem install bundler

COPY Gemfile Gemfile.lock* *.gemspec ./
COPY package.json package-lock.json* ./

RUN bundle install
RUN npm install

RUN git config --global --add safe.directory /site

EXPOSE 4000

ENTRYPOINT ["/bin/bash"]