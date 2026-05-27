# FROM swr.cn-north-4.myhuaweicloud.com/ci_cann/ubuntu22.04_x86:9.0.0-beta.1-910b-py3.11
FROM swr.cn-north-4.myhuaweicloud.com/ci_cann/ubuntu22.04_arm:9.0.0-beta.1-910b-py3.11


RUN mkdir /root/.pip \
    && echo "[global]" > /root/.pip/pip.conf \
    && echo "index-url=https://repo.huaweicloud.com/repository/pypi/simple" >> /root/.pip/pip.conf \
    && echo "trusted-host=repo.huaweicloud.com" >> /root/.pip/pip.conf \
    && echo "timeout=120" >> /root/.pip/pip.conf

RUN pip3 install esdk-obs-python --trusted-host mirrors.huaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple

COPY requirements.txt /tmp/requirements.txt
COPY requirements_dev.txt /tmp/requirements_dev.txt

RUN python3 -m pip install --no-cache-dir --prefer-binary --retries 10 --timeout 120 -r /tmp/requirements.txt \
    && python3 -m pip install --no-cache-dir --prefer-binary --retries 10 --timeout 120 -r /tmp/requirements_dev.txt

COPY ./cluster_smoke_task.sh /home/cluster_smoke_task.sh
COPY ./upload.py /home/upload.py
