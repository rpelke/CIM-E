## Push new docker image to dockerhub

1. Build image

    ```bash
    docker build -f docker/build_simulator.dockerfile -t cim-e .
    ```

2. Push image   
    ```bash
    docker login docker.io
    docker tag cim-e docker.io/pelke/cim-e:<tag>
    docker push docker.io/pelke/cim-e:<tag>
    ```
