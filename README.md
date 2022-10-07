# parrot_exps
Model training and experiments for Parrot


## Useful Commands

The following have been tested on the Lambda Labs GPU.

To start and enter the Docker container: 
```
chmod +x run_docker.sh
./run_docker.sh
```

The conda environment `base` will be enabled when you enter the container.

The script will run in such a way that changes that are made to the `parrot_exps` directory within the container will be visible as changes to `/home/team_030/parrot_exps` outside the container. (This was done using the 'bind mount' functionality.)

To re-make the Docker image from Dockerfile: `sudo docker image build -t pexps:latest .`


