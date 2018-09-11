FROM python:3.6.4

WORKDIR /root
RUN mkdir /root/output

COPY requirements.txt /root/
RUN pip install -r requirements.txt

COPY query_model.py working_data.py /root/
COPY output/ /root/output/

CMD python query_model.py --model output/model.mdl --encoder output/encoder.pck --sequence MSAAVKDER