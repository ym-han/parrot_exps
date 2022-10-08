# parrot_exps
Model training and experiments for Parrot


## Useful Commands

The following have been tested on the Lambda Labs GPU provided as part of the FSDL class.

**To start and enter the Docker container:**
```
chmod +x run_docker.sh
./run_docker.sh
```

Note that `run_docker.sh` uses `sudo` --- you should change that if you don't want that.

The conda environment `base` will be enabled when you enter the container.

`run_docker.sh` will run the container in such a way that changes that are made to the `parrot_exps` directory within the container will be visible as changes to `/home/team_030/parrot_exps` *outside* the container. (This was done using the 'bind mount' functionality.)

**To re-make the Docker image from Dockerfile:** `sudo docker image build -t pexps:latest .`


