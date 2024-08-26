# Etapa 1: Construcción
FROM node:20 AS build

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de configuración y las dependencias
COPY package.json package-lock.json tsconfig.json webpack.config.js /app
COPY ./dist/styles.css /app/dist/styles.css

# Instalar dependencias
RUN npm install

# Copiar el código fuente
COPY ./src /app/src
COPY ./index.html /app/index.html

# Compilar el código TypeScript y construir el bundle con webpack
RUN npm run build

# Etapa 2: Servir la aplicación
FROM node:20

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el build desde la etapa de construcción
COPY --from=build /app/dist /app/dist
COPY --from=build /app/index.html /app/index.html
COPY --from=build /app/package.json /app/package.json
COPY --from=build /app/package-lock.json /app/package-lock.json

# Instalar http-server
RUN npm install -g http-server

# Exponer el puerto 3000
EXPOSE 3000

# Comando para iniciar el servidor
CMD ["npm", "start"]
