# Base image with Node.js
FROM node:20-alpine AS build

# Set working directory
WORKDIR /app

# Set environment variable to indicate Docker build
ENV DOCKER_BUILD=true

# Copy package.json and package-lock.json
COPY package*.json ./

# Copy scripts directory for postinstall script
COPY scripts/ ./scripts/

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Create the build
RUN npm run build

# Production environment
FROM nginx:alpine

# Copy the build output to replace the default nginx contents
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose port 80
EXPOSE 80

# Entrypoint script for runtime configuration
ENTRYPOINT ["/docker-entrypoint.sh"]

# Command to run the server
CMD ["nginx", "-g", "daemon off;"]
