# build: docker build -t langchain_app .
# run: docker run -it --rm -p 8080:8080 langchain_app
# docker run -it --rm -p 8080:8080 -v /home/freinque/pycharm_projects/xray_report_gen/data/:/xray_report_gen/data langchain_app
FROM pytorch/pytorch
LABEL authors="freinque"

RUN pip install --upgrade pip
RUN pip install uvicorn
RUN pip install langchain openai huggingface_hub
RUN pip install "fastapi[standard]"
RUN pip install "transformers[torch]"
RUN pip install langchain-community
RUN pip install pandas
RUN pip install --default-timeout=1000 scikit-learn
RUN pip install tiktoken
RUN pip install einops torchvision
RUN pip install PIL

RUN mkdir /xray_report_gen
COPY ./ /xray_report_gen
COPY ./oa_token.txt /xray_report_gen/
COPY ./hf_token.txt /xray_report_gen/

ENV PYTHONPATH="[/usr/local/lib/python311.zip:/usr/local/lib/python3.11:/usr/local/lib/python3.11/lib-dynload:/usr/local/lib/python3.11/site-packages']"
ENV PYTHONPATH="$PYTHONPATH:/xray_report_gen"

WORKDIR /xray_report_gen/scripts
EXPOSE 8080

CMD ["uvicorn", "langchain_app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "debug"]
#CMD ["/bin/bash"]
