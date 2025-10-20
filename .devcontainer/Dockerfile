FROM mambaorg/micromamba:2.0.5-ubuntu20.04 as micromamba
# Using a rocker image as a base, as getting R packages installed is too much
# work compared to conda
FROM rocker/tidyverse:4.4
LABEL authors="apduncan"
LABEL org.opencontainers.image.source=https://github.com/apduncan/lordist_ctf_comparison

# Install anything possible from CRAN
RUN install2.r --error \
    devtools \
    ggplot2 \
    dplyr \
    readr \
    logger

RUN R -e "devtools::install_github('https://github.com/XinheQi/LorDist')"

# Add micromamba functionality
# see : https://micromamba-docker.readthedocs.io/en/latest/
#       advanced_usage.html#adding-micromamba-to-an-existing-docker-image
USER root

# if your image defaults to a non-root user, then you may want to make the
# next 3 ARG commands match the values in your image. You can get the values
# by running: docker run --rm -it my/image id -a
ARG MAMBA_USER=mambauser
ARG MAMBA_USER_ID=57439
ARG MAMBA_USER_GID=57439
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
# Optional: if you want to customize the ENTRYPOINT and have a conda
# environment activated, then do this:
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "my_entrypoint_program"]

# You can modify the CMD statement as needed....
CMD ["/bin/bash"]

COPY --chown=$MAMBA_USER:$MAMBA_USER gemelli_env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
RUN micromamba run -n base pip install cython==0.29.37
RUN micromamba run -n base pip install numpy==1.24.4
RUN micromamba run -n base pip install scikit-bio==0.6.2
RUN micromamba run -n base pip install gemelli

# Lazy - run as root
USER root