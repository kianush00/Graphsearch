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

# Etapa 2: Servir la aplicación con NGINX
FROM nginx:stable

# Eliminar configuración predeterminada de NGINX
RUN rm /etc/nginx/conf.d/default.conf

# Copiar el archivo de configuración de NGINX personalizado
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copiar los archivos compilados al directorio estático de NGINX
COPY --from=build /app/dist /usr/share/nginx/html/dist
COPY --from=build /app/index.html /usr/share/nginx/html

# Exponer el puerto 80 para NGINX
EXPOSE 80
