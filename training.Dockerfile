FROM python:3.6.4

WORKDIR /root
RUN mkdir /root/output

COPY requirements.txt /root/
RUN pip install -r requirements.txt

COPY grid_search_working_example.py neural_net.py nn_utils.py working_data.py bdata.20130222.mhci.txt /root/

CMD  python grid_search_working_example.py