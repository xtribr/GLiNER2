FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY enem-analytics/frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy source
COPY enem-analytics/frontend/ .

# Build
RUN npm run build

# Production image
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# Copy built files
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["node", "server.js"]
