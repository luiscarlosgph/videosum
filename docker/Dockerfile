# @brief  Image with support for: CUDA runtime and pyenv.
# @author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   27 September 2022.

# Image info
FROM nvidia/cuda:10.2-cudnn8-runtime
LABEL maintainer="luiscarlos.gph@gmail.com"
LABEL version="0.1"
LABEL description="Image with support for: CUDA runtime and pyenv."

# Install NVIDIA repository keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 134
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install commonly used packages
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y apt-utils aptitude sudo vim zsh wget curl git net-tools tmux locales build-essential cmake build-essential zlib1g-dev libbz2-dev libreadline-dev libssl-dev libsqlite3-dev libffi-dev liblzma-dev tk-dev ncurses-dev

# Install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install SSH server
RUN DEBIAN_FRONTEND=noninteractive apt install -y openssh-server

# Open ports
EXPOSE 22

# Add user 'docker' with password 'docker' and give sudo powers
ENV USER="docker"
ENV PASSWD="docker"
ENV HOME="/home/$USER"
ADD files/uid.txt /tmp/uid.txt
RUN DEBIAN_FRONTEND=noninteractive apt install -y python3 python3-passlib
RUN groupadd -g $(cat /tmp/uid.txt) $USER
RUN useradd -m -u $(cat /tmp/uid.txt) -g $(cat /tmp/uid.txt) -p "$(python3 -c 'import os; import passlib.hash; print(passlib.hash.sha512_crypt.hash(os.getenv("PASSWD")))')" -s /bin/zsh $USER
RUN usermod -a -G sudo $USER
RUN touch $HOME/.zshrc && chown $USER:$USER $HOME/.zshrc

# Intall oh-my-zsh so we have a nice terminal
RUN sudo -u $USER wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O $HOME/install.sh
RUN sudo -u $USER chmod +x $HOME/install.sh 
RUN sudo -u $USER $HOME/install.sh --unattended
RUN sudo -u $USER sed -i 's/robbyrussell/terminalparty/g' $HOME/.zshrc

# Add CUDA environment variables to the 'docker' user
RUN echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> $HOME/.zshrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' >> $HOME/.zshrc

# Install Python's pyenv environment
ENV PYTHON_VERSION="3.8.6"
RUN sudo -u $USER git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
RUN sudo -u $USER echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $HOME/.zshrc
RUN sudo -u $USER echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> $HOME/.zshrc
RUN sudo -u $USER echo 'eval "$(pyenv init --path)"' >> $HOME/.zshrc
RUN sudo -u $USER /bin/zsh -c "source $HOME/.zshrc && pyenv install $PYTHON_VERSION"
RUN sudo -u $USER /bin/zsh -c "source $HOME/.zshrc && pyenv global $PYTHON_VERSION"

# Start services and run the user-specified command
ADD files/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["sleep infinity"]
