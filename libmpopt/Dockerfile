FROM debian:bullseye

RUN printf '%s\n' \
    'Acquire::Retries "10";' \
    'Acquire::http::Timeout "120";' \
    'Acquire::https::Timeout "120";' \
    > /etc/apt/apt.conf.d/80-retries

RUN printf '%s\n' \
    'deb http://deb.debian.org/debian bullseye main' \
    'deb http://deb.debian.org/debian bullseye-updates main' \
    'deb http://security.debian.org/debian-security bullseye-security main' \
    > /etc/apt/sources.list

RUN export DEBIAN_FRONTEND="noninteractive" \
&& apt-get update \
&& apt-get install -y --no-install-recommends ca-certificates \
&& rm -rf /var/lib/apt/lists/*

RUN printf '%s\n' \
    'deb https://deb.debian.org/debian bullseye main' \
    'deb https://deb.debian.org/debian bullseye-updates main' \
    'deb https://security.debian.org/debian-security bullseye-security main' \
    > /etc/apt/sources.list

RUN export DEBIAN_FRONTEND="noninteractive" \
&& apt-get update \
&& apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    meson \
    ninja-build \
    pkg-config \
    python3 \
    python3-dev \
    python3-numpy \
    python3-pip \
    sudo \
    swig \
    unzip \
&& rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# In Debian the sysconfig Python module is reporting the wrong paths. Instead
# of site-packages, Debian uses dist-packages. We fix this by creating a
# symlink here.
RUN cd /usr/local/lib/python3.* \
&& ln -nsf dist-packages site-packages

RUN useradd -ms /bin/bash user \
&& echo 'user ALL=(ALL) NOPASSWD: ALL' >/etc/sudoers.d/user

USER user
WORKDIR /home/user

RUN mkdir worms \
&& cd worms \
&& curl -L -o worms.zip 'https://research-explorer.app.ist.ac.at/download/5561/5614/IST-2017-57-v1%2B1_wormMatchingProblems.zip' \
&& unzip worms.zip \
&& rm worms.zip *.h5

COPY --chown=user:user libqpbo libqpbo

RUN mkdir libqpbo-build \
&& cd libqpbo \
&& ./prepare \
&& cd ../libqpbo-build \
&& meson setup -Db_ndebug=if-release -Dbuildtype=release ../libqpbo \
&& ninja \
&& sudo meson install \
&& sudo ldconfig

COPY --chown=user:user qapopt qapopt

RUN mkdir qapopt-build \
&& cd qapopt-build \
&& meson setup \
    -Db_ndebug=if-release \
    -Dbuildtype=release \
    -Dqpbo=enabled \
    -Dgurobi=auto \
    /home/user/qapopt \
&& ninja \
&& sudo meson install \
&& sudo ldconfig

# vim: set ts=8 sts=4 sw=4 et ft=dockerfile:
