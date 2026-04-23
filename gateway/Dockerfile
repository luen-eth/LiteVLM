FROM node:20-alpine

WORKDIR /app

ENV NODE_ENV=production

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY src ./src
COPY auth-tokens.json ./auth-tokens.json

EXPOSE 8000

CMD ["node", "src/api.js"]
