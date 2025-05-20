FROM idm-docker-staging.packages.idmod.org/laser/laser-polio-base:latest

RUN pip3 install pipdeptree && pipdeptree -p laser-polio > /app/laser_polio_deps.txt

# Copy your application and data files (same as before)
COPY calib/ .
COPY calib/calib_configs/ ./calib/calib_configs/
COPY calib/model_configs/ ./calib/model_configs/
RUN chmod a+x cloud/check_study.sh
COPY ./data/ /app/data/

ENV POLIO_ROOT=/app
ENV NUMBA_CPU_NAME=generic
ENV HEADLESS=1

RUN pip3 install https://github.com/InstituteforDiseaseModeling/laser-polio/releases/download/v0.1.14/laser_polio-0.1.14-py3-none-any.whl

# Final cleanup
RUN pip3 cache purge

# Entrypoint
ENTRYPOINT ["python3", "calibrate.py"]
