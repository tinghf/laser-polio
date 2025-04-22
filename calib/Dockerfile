# Start with Rocky Linux 9
FROM rockylinux:9.3

# Enable IUS and EPEL
RUN dnf install -y epel-release && \
    dnf update -y

# Install Python 3.12 and tools from IUS
RUN dnf install -y \
    python3.12 \
    python3.12-devel \
    python3.12-pip \
    mariadb-connector-c-devel \
    gcc \
    gcc-c++ \
    make \
    wget \
    screen \
    procps \
    vim && \
    dnf clean all

# Set default python3 and pip3 to use 3.10 explicitly
RUN alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 && \
    alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.12 100

# Confirm
RUN python3 --version && pip3 --version

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --pre -i https://packages.idmod.org/api/pypi/pypi-production/simple laser-polio
RUN pip3 install optuna mysqlclient
RUN pip3 install pipdeptree && pipdeptree -p laser-polio > /app/laser_polio_deps.txt

# Copy your application and data files (same as before)
COPY calib/calib_db.py .
COPY calib/calibrate.py .
COPY calib/objective.py .
COPY calib/report.py .
COPY calib/scoring.py .
COPY calib/targets.py .
COPY calib/worker.py .
COPY calib/cloud/check_study.sh .
COPY calib/calib_configs/ ./calib/calib_configs/
COPY calib/model_configs/ ./calib/model_configs/
COPY data/node_lookup.json /app/data/node_lookup.json
RUN chmod a+x ./check_study.sh
COPY data/init_immunity_0.5coverage_january.h5 /app/data/
COPY data/distance_matrix_africa_adm2.h5 /app/data/
COPY ./data/Nigeria_age_pyramid_2024.csv /app/data/
COPY ./data/sia_scenario_1.csv /app/data/
COPY ./data/compiled_cbr_pop_ri_sia_underwt_africa.csv /app/data/
COPY ./data/age_africa.csv /app/data/
COPY ./data/sia_historic_schedule.csv /app/data/
COPY ./data/epi_africa_20250421.h5 /app/data/epi_africa_20250421.h5

ENV POLIO_ROOT=/app
ENV NUMBA_CPU_NAME=generic
ENV HEADLESS=1

# Final cleanup
RUN pip3 cache purge

# Entrypoint
ENTRYPOINT ["python3", "calibrate.py"]
