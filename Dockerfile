# Start with the official Node LTS image (Debian-based)
FROM node:22-bookworm

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install Ruby and build dependencies for Jekyll gems
RUN apt-get update && apt-get install -y \
    ruby-full \
    build-essential \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /site

# Install Bundler (Ruby's package manager)
RUN gem install bundler

# Copy dependency files first
COPY Gemfile Gemfile.lock* *.gemspec ./
COPY package.json package-lock.json* ./

# Install both Ruby and NPM dependencies
RUN bundle install
RUN npm install

EXPOSE 4000

# Keep the entrypoint as bash for easy debugging as requested
ENTRYPOINT ["/bin/bash"]