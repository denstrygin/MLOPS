FROM mambaorg/micromamba:1.5.6

WORKDIR /app

COPY conda.yaml .
RUN micromamba create -y -f conda.yaml && micromamba clean -a -y

COPY . .

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV CONDA_DEFAULT_ENV=mlops
ENV PATH /opt/conda/envs/mlops/bin:$PATH

EXPOSE 5000

ENTRYPOINT ["mlflow", "ui", "--host", "0.0.0.0"]